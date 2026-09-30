# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.utils.model_zoo as model_zoo
import pdb
from torchvision.models.resnet import BasicBlock, Bottleneck
# from torchvision.models.utils import load_state_dict_from_url
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, dropout=0.0):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.dropout = dropout

    def forward(self, input_image):
        self.features = []
        # x = (input_image - 0.45) / 0.225
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        # self.features.append(self.encoder.layer2(self.features[-1]))
        # self.features.append(self.encoder.layer3(self.features[-1]))
        # self.features.append(self.encoder.layer4(self.features[-1]))
        self.features.append(
            F.dropout(
                self.encoder.layer1(self.encoder.maxpool(self.features[-1])),
                p=self.dropout, training=True))
        self.features.append(
            F.dropout(
                self.encoder.layer2(self.features[-1]),
                p=self.dropout, training=True))
        self.features.append(
            F.dropout(self.encoder.layer3(self.features[-1]),
                p=self.dropout, training=True))
        self.features.append(
            F.dropout(
                self.encoder.layer4(self.features[-1]),
                p=self.dropout, training=True))

        return self.features

import sys
from typing import Type, Any, Callable, Union, List, Optional

def generate_binary_permutations(length):
    # Calculate the number of permutations
    num_permutations = 2**length
    
    # Create a tensor of all possible numbers from 0 to 2^length - 1
    numbers = torch.arange(num_permutations, dtype=torch.long)
    
    # Convert each number to its binary representation
    binary_tensor = torch.zeros(num_permutations, length, dtype=torch.long)
    for i in range(length):
        binary_tensor[:, length - 1 - i] = (numbers >> i) & 1
    
    return binary_tensor

class ResNetSkip(models.ResNet):
    def __init__(self, block, 
                 layers: List[int],
                 **kwargs: Any
                 ) -> None:
        self.layer_id = 1
        self.height = None
        self.width = None
        self.use_stereo = None
        self.num_layers = layers
        # do not mask the first layer
        self.mask_vectors = generate_binary_permutations(np.sum(layers) - 1)
        self.num_of_mask_combinations = self.mask_vectors.shape[0] 
        # no mask (all ones)
        self.mask_vector = self.mask_vectors[-1]
        super().__init__(block, layers, **kwargs)

    def set_mask_combination(self, idx):    
        self.mask_vector = self.mask_vectors[idx]

    def _make_layer( #self, block, planes, blocks, stride, dilate, group_id=1):
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False
    ):
        """ Create the whole group """
        ds_idx = 0
        for i in range(blocks):
            if self.layer_id > 1 and i == 0:
                stride = 2
            else:
                stride = 1

            meta = self._make_single_layer(block, planes, stride=stride, dilate=dilate)

            setattr(self, 'layer{}.{}'.format(self.layer_id, i), meta[1])
            if meta[0] is not None:
                setattr(self, 'layer{}.{}.downsample'.format(self.layer_id, i), meta[0])
                ds_idx += 1
        self.layer_id += 1
        
        if self.layer_id - 1 == 1:
            return self.layer1
        elif self.layer_id - 1 == 2:
            return self.layer2
        elif self.layer_id - 1 == 3:
            return self.layer3
        elif self.layer_id - 1 == 4:
            return self.layer4
 
    def _make_single_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        # layers = []
        # layers.append(
        layer = block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )

        self.inplanes = planes * block.expansion
        # for _ in range(1, blocks):
        #     layers.append(
        #         block(
        #             self.inplanes,
        #             planes,
        #             groups=self.groups,
        #             base_width=self.base_width,
        #             dilation=self.dilation,
        #             norm_layer=norm_layer,
        #         )
        #     )

        # return nn.Sequential(*layers)
        return downsample, layer

    def layer1(self, x: torch.Tensor):
        # TODO: write those more sensibly
        
        g = 0
        x = getattr(self, 'layer1.0')(x)
        self.prev = x  # input of next layer
        
        # 1st layer not masked
        mask_start_idx = int(np.sum([self.num_layers[:g]]) - int(g != 0))

        for i in range(1, self.num_layers[g]):
            if getattr(self, 'layer{}.{}.downsample'.format(g+1, i), None) is not None:
                self.prev = getattr(self, 'layer{}.{}.downsample'.format(g+1, i))(self.prev)
            x = getattr(self, 'layer{}.{}'.format(g+1, i))(x)
            
             
            mask = torch.full_like(x, self.mask_vector[mask_start_idx + i])
            self.prev = x = mask * x + (1-mask) * self.prev
        
        return x
    
    
    def layer2(self, x: torch.Tensor):
        g = 1

        # 1st layer not masked
        mask_start_idx = int(np.sum([self.num_layers[:g]]) - int(g != 0))

        for i in range(0 + int(g == 0), self.num_layers[g]):
            if getattr(self, 'layer{}.{}.downsample'.format(g+1, i), None) is not None:
                self.prev = getattr(self, 'layer{}.{}.downsample'.format(g+1, i))(self.prev)
            x = getattr(self, 'layer{}.{}'.format(g+1, i))(x)
        
            mask = torch.full_like(x, self.mask_vector[mask_start_idx + i])
            self.prev = x = mask * x + (1-mask) * self.prev
        return x

    def layer3(self, x: torch.Tensor):
        g = 2

        # 1st layer not masked
        mask_start_idx = int(np.sum([self.num_layers[:g]]) - int(g != 0))

        for i in range(0 + int(g == 0), self.num_layers[g]):
            if getattr(self, 'layer{}.{}.downsample'.format(g+1, i), None) is not None:
                self.prev = getattr(self, 'layer{}.{}.downsample'.format(g+1, i))(self.prev)
            x = getattr(self, 'layer{}.{}'.format(g+1, i))(x)
            
            mask = torch.full_like(x, self.mask_vector[mask_start_idx + i])
            self.prev = x = mask * x + (1-mask) * self.prev
        
        return x
    
    def layer4(self, x: torch.Tensor):
        g = 3

        # 1st layer not masked
        mask_start_idx = int(np.sum([self.num_layers[:g]]) - int(g != 0))
        
        for i in range(0 + int(g == 0), self.num_layers[g]):
            if getattr(self, 'layer{}.{}.downsample'.format(g+1, i), None) is not None:
                self.prev = getattr(self, 'layer{}.{}.downsample'.format(g+1, i))(self.prev)
            x = getattr(self, 'layer{}.{}'.format(g+1, i))(x)
        
            mask = torch.full_like(x, self.mask_vector[mask_start_idx + i])
            self.prev = x = mask * x + (1-mask) * self.prev
        return x

    # def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
    #     # See note [TorchScript super()]
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)

    #     # x = self.layer1(x)
    #     # x = self.layer2(x)
    #     # x = self.layer3(x)
    #     # x = self.layer4(x)


    #     # masks = []
    #     # gprobs = []
    #     # must pass through the first layer in first group
    #     x = getattr(self, 'layer1.0')(x)
    #     # gate takes the output of the current layer
    #     # gate_feature = getattr(self, 'group1_gate0')(x)
    #     # mask, gprob = self.control(gate_feature)
    #     # gprobs.append(gprob)
    #     # masks.append(mask.squeeze())
    #     prev = x  # input of next layer

    #     for g in range(4):
    #         for i in range(0 + int(g == 0), self.num_layers[g]):
    #             if getattr(self, 'layer{}.{}.downsample'.format(g+1, i)) is not None:
    #                 prev = getattr(self, 'layer{}.{}.downsample'.format(g+1, i))(prev)
    #             x = getattr(self, 'layer{}.{}'.format(g+1, i))(x)
    #             # prev = x = mask.expand_as(x)*x + (1-mask).expand_as(prev)*prev
    #             # gate_feature = getattr(self, 'group{}_gate{}'.format(g+1, i))(x)
    #             # mask, gprob = self.control(gate_feature)
    #             # if not (g == 3 and i == (self.num_layers[3]-1)):
    #             #     # not add the last mask to masks
    #             #     gprobs.append(gprob)
    #             #     masks.append(mask.squeeze())


    #     x = self.avgpool(x)
    #     x = torch.flatten(x, 1)
    #     x = self.fc(x)

    #     return x

def skip_resnet(
    arch: str,
    block: Type[Union[models.resnet.BasicBlock, models.resnet.Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNetSkip:
    model = ResNetSkip(block, layers, **kwargs)
    if pretrained:
        state_dict =  load_state_dict_from_url(models.resnet.model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

# # overwrite _resnet function for it to output my version of ResNet with layer skips
# models.resnet._resnet = _skip_resnet