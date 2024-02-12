import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
import cv2

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead

class DeepLabV3plus(smp.DeepLabV3Plus):
    def __init__(self, **kwargs):
        super().__init__(kwargs['encoder_name'])
    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        logit = self.segmentation_head(decoder_output)
        return {'seg_logits':logit,'encoder_features':features[1:],'decoder_features':decoder_output}