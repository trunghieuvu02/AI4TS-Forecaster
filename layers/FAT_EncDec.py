import torch
import torch.nn as nn
from typing import List, Tuple, Any
from layers.Autoformer_EncDec import series_decomp
from layers.SelfAttention_Family import MultiHead_LogSparse_Attention, MultiHead_AutoCorrelation_Attention
from torch import Tensor
import torch.nn.functional as F


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Note that delta only used for Self-Attention(x_enc with x_enc)
        # and Cross-Attention(x_enc with x_dec),
        # but not suitable for Self-Attention(x_dec with x_dec)
        # x = [32, 144, 512], cross = [32, 96, 512], x_mask = None, cross_mask = None
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class FeedForward(nn.Module):
    r"""Feed Forward Block

    Args:
        d_model (`int`):
            Dimension of expected input.
        d_ff (`int`, default to 512):
            Dimension of feed forward layer (default=512).
        dropout (`float`, default to 0.1):
            Dropout value (default=0.1).
    """

    def __init__(self, d_model: int, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        # set d_ff as a default to 512
        self.linear_1 = nn.Linear(int(d_model / 2), d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, int(d_model / 2))

    def forward(self, x: Tensor):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class AttentionLayer(nn.Module):
    def __init__(self, multi_attention, input_dim: int, output_dim: int, d_model_type: int, n_heads_type: int,
                 attention_type: str,
                 causal_kernel_size: int = 3, value_kernel_size: int = 1, dropout: float = 0.1,
                 auto_moving_avg: int = 25) -> None:
        super(AttentionLayer, self).__init__()

        self.d_model_type = d_model_type
        self.n_heads_type = n_heads_type

        self.attention_type = attention_type

        # select the current attention type
        self.multi_attention = multi_attention

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        # the dimensions of key and value in each head
        self.d_head_key = int(d_model_type / n_heads_type)
        self.d_head_value = int(d_model_type / n_heads_type)

        # Use Conv-1D to extract high-dimensional features
        self.causal_kernel_size = causal_kernel_size
        self.value_kernel_size = value_kernel_size

        self.query_projection = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.n_heads_type * self.d_head_key,
            kernel_size=self.causal_kernel_size,
        )

        self.key_projection = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.n_heads_type * self.d_head_key,
            kernel_size=self.causal_kernel_size,
        )

        self.value_projection = nn.Conv1d(
            in_channels=input_dim,
            out_channels=self.n_heads_type * self.d_head_key,
            kernel_size=self.value_kernel_size
        )

        # out_projection
        self.out_projection = nn.Conv1d(in_channels=self.d_head_value * self.n_heads_type,
                                        out_channels=output_dim,
                                        kernel_size=self.value_kernel_size)

        if self.attention_type == "auto":
            self.decomp_block = series_decomp(auto_moving_avg)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor):
        r"""Pass the queries, keys, and values through the multi-head attention.

        Args:
            queries: the query matrix.
            keys: the key matrix.
            values: the value matrix.

        Shape:
            queries shape is (batch_size, length_Q, d_model_Q)
            keys shape is (batch_size, length_K, d_model_K)
            values shape is (batch_size, length_V, d_model_V)
        """

        # Get shape of queries, keys, values
        batch_size, length_Q, d_model_Q = queries.shape
        _, length_K, d_model_K = keys.shape
        _, length_V, d_model_V = values.shape

        # Use 1d-CNN to extract high-dimensional features
        queries_padding_size = int(self.causal_kernel_size / 2)
        keys_padding_size = int(self.causal_kernel_size / 2)
        values_padding_size = int(self.value_kernel_size / 2)

        # Padding is to preserve the input dimension after using CNN
        queries_padding = nn.functional.pad(
            queries.permute(0, 2, 1),  # (batch_size, d_model/channel, seq_length)
            pad=(queries_padding_size, queries_padding_size),
            mode='replicate'
        )
        keys_padding = nn.functional.pad(
            keys.permute(0, 2, 1),  # (batch_size, d_model/channel, seq_length)
            pad=(keys_padding_size, keys_padding_size),
            mode='replicate'
        )
        values_padding = nn.functional.pad(
            values.permute(0, 2, 1),  # (batch_size, d_model/channel, seq_length)
            pad=(values_padding_size, values_padding_size),
            mode='replicate'
        )

        # Transform to CNN
        queries = self.query_projection(queries_padding).permute(0, 2, 1)  # (batch_size, seq_length, d_model/channel)
        keys = self.key_projection(keys_padding).permute(0, 2, 1)  # (batch_size, seq_length, d_model/channel)
        values = self.value_projection(values_padding).permute(0, 2, 1)  # (batch_size, seq_length, d_model/channel)

        # Get each of head
        head_v = self.n_heads_type

        # Convert Q, K, V data into the shape of [bs, length, n_heads, head_v]
        query = queries.view(batch_size, length_Q, head_v, -1)
        key = keys.view(batch_size, length_K, head_v, -1)
        value = values.view(batch_size, length_V, head_v, -1)

        # Calculate multi-headed attention types
        out, attn_score = self.multi_attention(query, key, value)

        # Transpose the fixed format of out [bs, length, d_model]
        out = out.view(batch_size, length_Q, -1)
        padding_out = nn.functional.pad(out.permute(0, 2, 1),
                                        pad=(values_padding_size, values_padding_size),
                                        mode='replicate')

        out = self.activation(self.out_projection(padding_out)).permute(0, 2, 1)
        out = self.dropout(out)
        # print(f"=================> {self.attention_type} === {out.shape}")
        # If it is auto
        if self.attention_type == "auto":
            out, _ = self.decomp_block(out)
            # print("====> out = ", out.shape)
            return out, attn_score
        else:
            return out, attn_score


class EncoderLayer(nn.Module):
    def __init__(self,
                 attention_layer_types: List,
                 n_heads_log: int,
                 n_heads_auto: int,
                 feedforward_dim: None,
                 d_model: int = 128,
                 n_layers: int = 2,
                 feat_dim: int = 14,
                 # device=None,
                 dropout: float = 0.1,
                 causal_kernel_size: int = 3,
                 value_kernel_size: int = 1,
                 auto_moving_avg: int = 25,
                 ) -> None:
        super(EncoderLayer, self).__init__()

        self.attention_layer_types = attention_layer_types

        self.n_layers = n_layers
        self.feat_dim = feat_dim
        self.feedforward_dim = feedforward_dim or int(d_model * 2)
        self.dropout = dropout

        # self.device = device

        # Calculation of d_model for each head
        n_heads = n_heads_log + n_heads_auto
        d_model_each_head = d_model // n_heads

        attentions = []

        for attention_type in self.attention_layer_types:
            if attention_type == "log":
                attentions.append(AttentionLayer(
                    multi_attention=MultiHead_LogSparse_Attention(dropout=0.1, mask_flag=True),
                    input_dim=d_model,
                    output_dim=d_model_each_head * n_heads_log,
                    d_model_type=d_model_each_head * n_heads_log,
                    n_heads_type=n_heads_log,
                    attention_type=attention_type,
                    dropout=dropout,
                    causal_kernel_size=causal_kernel_size,
                    value_kernel_size=value_kernel_size,
                    auto_moving_avg=auto_moving_avg
                ))
            elif attention_type == "auto":
                attentions.append(AttentionLayer(
                    multi_attention=MultiHead_AutoCorrelation_Attention(mask_flag=True,
                                                                        factor=5,
                                                                        scale=None,
                                                                        dropout=dropout,
                                                                        output_attention=True),
                    input_dim=d_model,
                    output_dim=d_model_each_head * n_heads_auto,
                    d_model_type=d_model_each_head * n_heads_auto,
                    n_heads_type=n_heads_auto,
                    attention_type=attention_type,
                    dropout=dropout,
                    causal_kernel_size=causal_kernel_size,
                    value_kernel_size=value_kernel_size,
                    auto_moving_avg=auto_moving_avg
                ))
            else:
                raise ValueError("Attention type not supported")

        self.attentions = nn.ModuleList(attentions)
        self.feedforward = FeedForward(d_model, d_ff=self.feedforward_dim)

        # layer norms
        self.layer_norm1 = nn.LayerNorm(int(d_model / 2))
        self.layer_norm2 = nn.LayerNorm(int(d_model / 2))

        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Only use with auto block
        self.decomp_block = series_decomp(auto_moving_avg)  # res, moving_mean

    def do_the_rest_log_encoder(self, conv_input, output_attention):
        # Create a copy of the src tensor
        # src_copy = src.clone()

        # Residual connection + Layer Norm 1
        conv_input = conv_input + self.dropout1(output_attention)
        conv_input = self.layer_norm1(conv_input)  # [batch_size, seq_length, d_head_log]

        # Feed forward block
        ffn_out = self.feedforward(conv_input)

        # Residual connection + LayerNorm2
        ffn_out = self.dropout2(ffn_out)
        out = self.layer_norm2(conv_input + ffn_out)  # [batch_size, seq_length, d_head_log]
        return out

    def do_the_rest_auto_encoder(self, conv_input, output_attention):
        # Create a copy of the src tensor
        # src_copy = src.clone()
        # Residual connection + Series Decompose
        conv_input = conv_input + self.dropout1(output_attention)
        conv_input, _ = self.decomp_block(conv_input)

        # Feed forward block
        ffn_out = self.feedforward(conv_input)

        # Residual connection + LayerNorm2
        ffn_out = self.dropout2(ffn_out)
        out, _ = self.decomp_block(conv_input + ffn_out)  # [batch_size, seq_length, d_head_log]
        return out

    def forward(self, src: Tensor, conv_input: Tensor) -> tuple[Any, list[Any]]:
        outs, attention_scores = [], []
        # attention_type in self.attention_layer_types
        for attention_layer, attention_type in zip(self.attentions, self.attention_layer_types):
            # do self-multi_head attention
            out, attn_score = attention_layer(src, src, src)
            if attention_type == "log":
                out = self.do_the_rest_log_encoder(conv_input, out)
            elif attention_type == "auto":
                out = self.do_the_rest_auto_encoder(conv_input, out)

            outs.append(out)
            attention_scores.append(attn_score)

        final_outs = torch.cat(outs, dim=-1)
        return final_outs, attention_scores


class Encoder(nn.Module):
    """
    Fusion Attention Transformer (FAT) Encoder
    """

    def __init__(self, encoder_layers):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(encoder_layers)

    def forward(self, src, conv_input):
        """
        Forward pass for the FAT Encoder
        :param src: input sequence
        :param conv_input: convolutional input
        :return: output of the encoder
        """
        is_batched = src.dim() == 3
        if not is_batched:
            raise RuntimeError(
                "the shape of src should be (batch_size, sequence_length, d_model). The system got {} dimensions".format(
                    src.dim()))

        attentions = []
        for encoder_layer in self.encoder_layers:
            src, attention = encoder_layer(src, conv_input)
            attentions.append(attention)

        return src, attentions
