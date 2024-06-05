# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py

# https://github.com/felixgwu/img_classification_pk_pytorch/blob/master/models/densenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
#from torchvision.models.densenet import _Transition
import math
import numpy as np



class _Transition(nn.Sequential):
    def __init__(self,num_input_features,num_output_features):
        super(_Transition,self).__init__()
        self.add_module('norm',nn.BatchNorm2d(num_input_features))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv',nn.Conv2d(num_input_features,num_output_features,
            kernel_size=1,stride=1,bias=False))
        self.add_module('pool',nn.AvgPool2d(kernel_size=2,stride=2))




class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),

        # If the bottle neck mode is set, apply feature reduction to limit the growth of features
        # Why should we expand the number of features by bn_size*growth?

        # https://stats.stackexchange.com/questions/194142/what-does-1x1-convolution-mean-in-a-neural-network
        if bn_size > 0:
            interChannels = 4*growth_rate
            self.add_module('conv1', nn.Conv2d(
                num_input_features, interChannels, kernel_size=1, stride=1, bias=False))
            self.add_module('norm2', nn.BatchNorm2d(interChannels))
            self.add_module('conv2', nn.Conv2d(
                interChannels, growth_rate, kernel_size=3, padding=1, bias=False))
        else:
            self.add_module('conv2', nn.Conv2d(
                num_input_features, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i *
                                growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i+1), layer)


class DenseNet(nn.Module):
    """Densenet-BC model class, based on
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes

        growth_rate=12, block_config=(16, 16, 16), compression=0.5,num_init_features=24
        , bn_size=4, drop_rate=0, avgpool_size=8,
    """
    def __init__(self, growth_rate=12, block_config=(16, 16, 16), compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0, avgpool_size=4):
        super(DenseNet, self).__init__()

        self.avgpool_size = avgpool_size
        # The first Convolution layer
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features,
                                kernel_size=3, stride=1, padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        # Did not add the pooling layer to preserve dimension
        # The number of layers in each Densnet is adjustable

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            Dense_block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                      bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            # Add name to the Denseblock
            self.features.add_module('denseblock%d' % (i + 1), Dense_block)

            # Increase the number of features by the growth rate times the number
            # of layers in each Denseblock
            num_features += num_layers * growth_rate

            # check whether the current block is the last block
            # Add a transition layer to all Denseblocks except the last
            if i != len(block_config):
                # Reduce the number of output features in the transition layer

                nOutChannels = int(math.floor(num_features*compression))

                trans = _Transition(num_input_features=num_features,
                                    num_output_features=nOutChannels)
                self.features.add_module('transition%d' % (i + 1), trans)
                # change the number of features for the next Dense block
                num_features = nOutChannels

            # Final batch norm
            self.features.add_module('last_norm%d' % (i+1), nn.BatchNorm2d(num_features))

        # Linear layer
        self.fc = nn.Linear(8208, 1024)
        self.fc_dist = nn.Linear(1024, 96)  # 12,12,32
        self.fc_intensity = nn.Linear(1024, 1)
        self.fc_rgb_ratio = nn.Linear(1024, 3)
        self.fc_ambient = nn.Linear(1024, 3)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)

    # x, input of shape (N=16, 3, 192, 256)
    def forward(self, x):
        features = self.features(x) # (16, 171, 24, 32)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1) # (16, 171 * 6 * 8)
        out = self.fc(out) # (16, 1024)

        dist_pred = self.fc_dist(out) # (16, 96)
        dist_pred = self.leaky_relu(dist_pred)
        dist_pred_min, _ = torch.min(dist_pred, dim=1, keepdim=True)
        dist_pred = dist_pred - dist_pred_min
        dist_pred_sum = torch.sum(dist_pred, axis=1).view(-1, 1)
        dist_pred = dist_pred / dist_pred_sum

        intenstiy_pred = self.fc_intensity(out) # (16, 1)
        # intenstiy_pred = self.relu(intenstiy_pred)

        rgb_ratio_pred = self.fc_rgb_ratio(out) # (16, 3)
        # rgb_ratio_pred = self.sigmoid(rgb_ratio_pred)

        ambient_pred = self.fc_ambient(out) # (16, 3)
        # ambient_pred = self.relu(ambient_pred)

        return {'distribution': dist_pred,
                'intensity': intenstiy_pred,
                'rgb_ratio': rgb_ratio_pred,
                'ambient': ambient_pred,
                }

class OriginalDenseNet(nn.Module):
    def __init__(self):
        super(OriginalDenseNet, self).__init__()

        self.features = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        # for param in self.features.parameters():
        #     param.require_grad = False
        H = 1024

        # Linear layer
        self.fc = nn.Linear(1000, H)
        self.fc_dist = nn.Linear(H, 96)  # 12,12,32
        self.fc_intensity = nn.Linear(H, 1)
        self.fc_rgb_ratio = nn.Linear(H, 3)
        self.fc_ambient = nn.Linear(H, 3)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.use_leaky_relu = False

    # x, input of shape (N=16, 3, 192, 256)
    def forward(self, x):
        features = self.features(x) # (16, 1000)
        out = F.relu(features, inplace=True)
        out = self.fc(out) # (16, 1024)

        dist_pred = self.fc_dist(out) # (16, 96)
        if self.use_leaky_relu:
            dist_pred = self.leaky_relu(dist_pred)
            dist_pred_min, _ = torch.min(dist_pred, dim=1, keepdim=True)
            dist_pred = dist_pred - dist_pred_min
            dist_pred_sum = torch.sum(dist_pred, axis=1).view(-1, 1)
            dist_pred = dist_pred / dist_pred_sum

        intenstiy_pred = self.fc_intensity(out) # (16, 1)

        rgb_ratio_pred = self.fc_rgb_ratio(out) # (16, 3)

        ambient_pred = self.fc_ambient(out) # (16, 3)

        return {'distribution': dist_pred,
                'intensity': intenstiy_pred,
                'rgb_ratio': rgb_ratio_pred,
                'ambient': ambient_pred,
                }

class IntensityNet(nn.Module):
    def __init__(self):
        super(IntensityNet, self).__init__()

        # self.max_pool = nn.MaxPool2d(4, stride=4) # (N, 3, 48, 64)
        self.avg_pool = nn.AvgPool2d(16, stride=16) # (N, 3, 12, 16)
        self.flatten = nn.Flatten() # (N, 3*12*16)
        self.fc = nn.Linear(3*12*16, 1)

    def forward(self, x):
        # out = self.max_pool(x)
        out = self.avg_pool(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out
