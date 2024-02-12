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

class DeepLabV3plus(smp.DeepLabV3Plus):
    def __init__(self, **kwargs):
        super().__init__(kwargs['encoder_name'])
    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        logit = self.segmentation_head(decoder_output)
        return {'seg_logits':logit,'encoder_features':features[1:],'decoder_features':decoder_output}


class BasicBlock(nn.Module):
    
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class AttentionGate(nn.Module):
    def __init__(self, in_channels):
        super(AttentionGate, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels+1),
            nn.Conv2d(in_channels+1, in_channels+1, 1),
            nn.ReLU(), 
            nn.Conv2d(in_channels+1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, input_features, gating_features):
        alpha = self.conv(torch.cat([input_features, gating_features], dim=1))
        output_features = (input_features * (alpha + 1)) 
        return output_features

class EdgeModule(nn.Module):
    def __init__(self,encoder_name='resnet50',with_grad=True,at_num=3):
        super().__init__()

        self.with_grad = with_grad
        if encoder_name=='resnet50':
            if at_num>3:
                self.dcn2 = nn.Conv2d(256, 1, 1)
            self.dcn3 = nn.Conv2d(512, 1, 1)
            self.dcn4 = nn.Conv2d(1024, 1, 1)
            self.dcn5 = nn.Conv2d(2048, 1, 1)
        elif encoder_name=='resnet18' or encoder_name== 'resnet34':
            if at_num>3:
                self.dcn2 = nn.Conv2d(64, 1, 1)
            self.dcn3 = nn.Conv2d(128, 1, 1)
            self.dcn4 = nn.Conv2d(256, 1, 1)
            self.dcn5 = nn.Conv2d(512, 1, 1)
        
        self.res1 = BasicBlock(64, 64, stride=1)
        self.ddc1 = nn.Conv2d(64, 32, 1)
        self.res2 = BasicBlock(32, 32, stride=1)
        self.ddc2 = nn.Conv2d(32, 16, 1)
        self.res3 = BasicBlock(16, 16, stride=1)
        self.ddc3 = nn.Conv2d(16, 8, 1)
        self.ddc4 = nn.Conv2d(8, 1, 1)
        self.gate1 = AttentionGate(32)
        self.gate2 = AttentionGate(16)
        self.gate3 = AttentionGate(8)
        
        self.fuse = nn.Conv2d(2, 1, 1)
        self.at_num = at_num

    def forward(self,feats,**kw):
        edge_feats = []
        x_size = feats[0].shape
        bm = self.res1(feats[1])
        bm = self.ddc1(bm) 
        bm = F.interpolate(bm,x_size[-2:],mode='bilinear',align_corners=True)
        edge_feats.append(bm)
        at_num = self.at_num
        assert at_num>0

        if at_num>3:
            at0 = self.dcn2(feats[2])
            at0 = F.interpolate(at0,x_size[-2:],mode='bilinear',align_corners=True)
            bm = self.gate1(bm,at0)
            edge_feats.append(bm)
        
        if at_num>2:
            at1 = self.dcn3(feats[3])
            at1 = F.interpolate(at1,x_size[-2:],mode='bilinear',align_corners=True)
            bm = self.gate1(bm,at1)
        edge_feats.append(bm)

        bm = self.res2(b)
        bm = self.ddc2(bm)

        if at_num>1:
            at2 = self.dcn4(feats[4])
            at2 = F.interpolate(at2,x_size[-2:],mode='bilinear',align_corners=True)
            bm = self.gate2(bm,at2)
        edge_feats.append(bm)

        bm = self.res3(bm)
        bm = self.ddc3(bm)

        if at_num==1:
            at3 = self.dcn5(feats[5])
            at3 = F.interpolate(at3,x_size[-2:],mode='bilinear',align_corners=True)
            bm = self.gate3(bm,at3)
        edge_feats.append(bm)

        bm = self.ddc4(bm)
        edge_logits = bm
        if self.with_grad:
            edge_map = kw['edge_map']
            edge_fuse = self.fuse(torch.cat([bm,edge_map],dim=1))
            edge_feats.append(edge_fuse)
        else:
            edge_fuse = edge_logits
        return edge_logits,edge_fuse,edge_feats

class BIDNet(SegmentationModel):
    def __init__(self,
        encoder_name: str = "resnet50",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        encoder_output_stride: int = 16,
        decoder_channels: int = 256,
        decoder_atrous_rates: tuple = (12, 24, 36),
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,
        upsampling: int = 4,
        with_grad=True,
        at_num=3,
        **kwargs) -> None:
        super().__init__()

        self.classes = classes
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride
        )
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )
        self.seg_head = SegmentationHead(decoder_channels+1, classes, activation=activation, upsampling=upsampling)
        
        self.edge_branch = EdgeModule(encoder_name,with_grad,at_num)
        self.fuse_conv = nn.Conv2d(2,1,1)

    def forward(self, x, edge):
        feats = self.encoder(x)
        decode = self.decoder(*feats)
        edge_logits,edge_fuse,edge_feats = self.edge_branch(feats,edge_map=edge)
        edge_fuse = F.interpolate(edge_fuse,decode.shape[-2:],mode='bilinear',align_corners=True)
        decode_fuse = torch.cat([decode,edge_fuse],dim=1)
        seg_logits = self.seg_head(decode_fuse)
        return {'seg_logits':seg_logits,'edge_logits':edge_logits,'encoder_features':feats[1:],'decoder_features':decode,'edge_features':edge_feats}
    

if __name__=='__main__':
    model = BIDNet()
    img = torch.randn(4, 3, 800, 800)
    output = model(img)
    pass