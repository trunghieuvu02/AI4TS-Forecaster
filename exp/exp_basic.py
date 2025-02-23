import os
import torch
import logging
from models import Autoformer, TimesNet, PatchTST, MICN, Crossformer, TimeMixer
from models import SegRNN, FAT

logger = logging.getLogger('__main__')


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'PatchTST': PatchTST,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'TimeMixer': TimeMixer,
            'SegRNN': SegRNN,
            'FAT': FAT
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

        # print(f"Total number of parameters size: {self.model.parameters()}")
        # total_params = sum(p.numel() for p in self.model.parameters())
        # param_size_bytes = total_params * 4  # 32-bit float
        # param_size_mb = param_size_bytes / (1024 ** 2)
        # print(f"Total number of parameters: {total_params}")
        # print(f"Model Size: {param_size_mb:.2f} MB")

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            logger.info('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            logger.info('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
