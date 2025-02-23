import torch
import torch.nn as nn

from layers.Transformer_EncDec import Decoder, DecoderLayer
from layers.FAT_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.n_heads_log = configs.n_heads_log
        self.n_heads_auto = configs.n_heads_auto
        self.attention_layer_types = configs.attention_layer_types
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.feat_dim = configs.enc_in
        self.feedforward_dim = configs.d_ff
        # self.device = configs.device
        self.causal_kernel_size = configs.causal_kernel_size
        self.value_kernel_size = configs.value_kernel_size
        self.auto_moving_avg = configs.auto_moving_avg

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention_layer_types=self.attention_layer_types,
                    d_model=self.d_model,
                    n_heads_log=self.n_heads_log,
                    n_heads_auto=self.n_heads_auto,
                    n_layers=self.e_layers,
                    feat_dim=self.feat_dim,
                    feedforward_dim=self.feedforward_dim,
                    dropout=configs.dropout,
                    # device=self.device,
                    causal_kernel_size=self.causal_kernel_size,
                    value_kernel_size=self.value_kernel_size,
                    auto_moving_avg=self.auto_moving_avg,
                )
                for layer_num in range(configs.e_layers)
            ]
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, dec_self_mask, dec_enc_mask):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        # Add data embedding + Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, conv_input=3)

        # Add data embedding + Decoder
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # De-normalization
        dec_out = dec_out * std_enc + mean_enc

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, dec_self_mask, dec_enc_mask)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
