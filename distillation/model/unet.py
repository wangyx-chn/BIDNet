import torch
from torch import nn
from typing import Optional, Union, List
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import cv2

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder

class UNet(smp.Unet):
    def __init__(self, **kwargs):
        super().__init__(kwargs['encoder_name'])
    def forward(self, x):
        features = self.encoder(x)
        # print(list(map(lambda x: x.shape, features)))
        # features[0]为输入
        decoder_output = self.decoder(*features)
        # Deeplabv3p 256通道的四倍下采样输出
        # print(decoder_output.shape) # .x256x128x128
        logit = self.segmentation_head(decoder_output)
        # 没有经过sigmoid的输出
        # 输出统一为元组
        return (logit,)