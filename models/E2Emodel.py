import os
import argparse
from contextlib import nullcontext
import random
import numpy as np
# from tabulate import tabulate
import torch

torch.backends.cudnn.benchmark = True
from models.common import  BaseRGBModel
from models.shift import make_temporal_shift
from models.modules import *
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from utils.misc import  seg_pool_2d
from models.PS_parts import inconv, down, double_conv

import requests
import urllib3


# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
#
# response = requests.get('https://127.0.0.1', verify=False)
# print(response.text)
# Prevent the GRU params from going too big (cap it at a RegNet-Y 800MF)
MAX_GRU_HIDDEN_DIM = 768

class C_channel(nn.Module):
    def __init__(self, fea_dim, hidden_dim=512, output_dim=64):
        super(C_channel, self).__init__()
        self.layer0 = nn.Linear(fea_dim, fea_dim+128)
        self.layer1 = nn.Linear(fea_dim+128, 256)
        # self.layer2 = nn.Linear(int((fea_dim+128)/2), 64)
        self.activation_1 = nn.ReLU()

    def forward(self, x):
        x3 = self.activation_1(self.layer0(x))
        x4 = self.activation_1(self.layer1(x3))
        # x5 = self.activation_1(self.layer2(x4))
        # return x5
        return x4

class E2EModel(nn.Module):

    def __init__(self, feature_arch, clip_len, modality):
        super().__init__()
        is_rgb = modality == 'rgb'
        in_channels = {'flow': 2, 'bw': 1, 'rgb': 3, 'pose': 16}[modality]

        if feature_arch.startswith(('rn18','rn34', 'rn50')):
            resnet_name = feature_arch.split('_')[0].replace('rn', 'resnet')
            features = getattr(
                torchvision.models, resnet_name)(pretrained=is_rgb)
            feat_dim = features.fc.in_features
            features.fc = nn.Identity()
            # import torchsummary
            # print(torchsummary.summary(features.to('cuda'), (3, 224, 224)))

            # Flow has only two input channels
            if not is_rgb:
                # FIXME: args maybe wrong for larger resnet
                features.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                    padding=(3, 3), bias=False)

        elif feature_arch.startswith(('rny002', 'rny008')):
            features = timm.create_model({
                                             'rny002': 'regnety_002',
                                             'rny008': 'regnety_008',
                                         }[feature_arch.rsplit('_', 1)[0]], pretrained=is_rgb)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()

            if not is_rgb:
                features.stem.conv = nn.Conv2d(
                    in_channels, 32, kernel_size=(3, 3), stride=(2, 2),
                    padding=(1, 1), bias=False)

        elif 'convnextt' in feature_arch:
            features = timm.create_model('convnext_tiny', pretrained=is_rgb)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()

            if not is_rgb:
                features.stem[0] = nn.Conv2d(
                    in_channels, 96, kernel_size=4, stride=4)

        else:
            raise NotImplementedError(feature_arch)

        # Add Temporal Shift Modules
        self._require_clip_len = -1
        if feature_arch.endswith('_tsm'):
            make_temporal_shift(features, clip_len, is_gsm=False)
            self._require_clip_len = clip_len
        elif feature_arch.endswith('_gsm'):
            make_temporal_shift(features, clip_len, is_gsm=True)
            self._require_clip_len = clip_len

        self._features = features
        self._feat_dim = feat_dim



    def forward(self, x):
        batch_size, channels, true_clip_len, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4)

        clip_len = true_clip_len
        if self._require_clip_len > 0:
            # TSM module requires clip len to be known
            assert true_clip_len <= self._require_clip_len, \
                'Expected {}, got {}'.format(
                    self._require_clip_len, true_clip_len)
            if true_clip_len < self._require_clip_len:
                x = F.pad(
                    x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                clip_len = self._require_clip_len

        # print(x.size()) ([16, 96, 3, 112, 112])
        im_featmap = self._features(
            x.reshape(-1, channels, height, width)
        )

        im_feat = im_featmap.reshape(batch_size, clip_len, self._feat_dim)
        if true_clip_len != clip_len:
            im_feat = im_feat[:, :true_clip_len, :]

        return im_feat


class SpotModel(nn.Module):
    def __init__(self, num_classes, temporal_arch, feat_dim = 368):
        super().__init__()

        if 'gru' in temporal_arch:  # default
            hidden_dim = feat_dim
            if hidden_dim > MAX_GRU_HIDDEN_DIM:
                hidden_dim = MAX_GRU_HIDDEN_DIM
                print('Clamped GRU hidden dim: {} -> {}'.format(
                    feat_dim, hidden_dim))
            if temporal_arch in ('gru', 'deeper_gru'):
                self._pred_fine = GRUPrediction(
                    feat_dim, num_classes, hidden_dim,
                    num_layers=3 if temporal_arch[0] == 'd' else 1)
            else:
                raise NotImplementedError(temporal_arch)
        elif temporal_arch == 'mstcn':
            self._pred_fine = TCNPrediction(feat_dim, num_classes, 3)
        elif temporal_arch == 'asformer':
            self._pred_fine = ASFormerPrediction(feat_dim, num_classes, 3)
        elif temporal_arch == '':
            self._pred_fine = FCPrediction(feat_dim, num_classes)
        else:
            raise NotImplementedError(temporal_arch)

    def forward(self,x):

        pred = self._pred_fine(x)
        # print(pred.size(),self._pred_fine)

        return pred


class EnhModel (nn.Module):
    def __init__(self,feat_dim, hidden_dim, num_layers=3):
        super().__init__()

        self.enhance = nn.GRU(feat_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self._dropout = nn.Dropout()

    def forward(self, x):
        # x: 8, dim, 96
        # (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1)
        # print(x.size())
        y, _ = self.enhance(x)
        return self._dropout(y)




