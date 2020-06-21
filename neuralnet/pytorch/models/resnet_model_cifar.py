import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .layers import DiagonalCirculantLayer

def conv3x3(in_planes, out_planes, stride=1):
  return nn.Conv2d(
    in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    init.xavier_uniform_(m.weight, gain=np.sqrt(2))
    init.constant_(m.bias, 0)


def cfg(depth):
  depth_lst = [18, 34, 50, 101, 152]
  assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
  cf_dict = {
    '18': (BasicBlock, [2, 2, 2, 2]),
    '34': (BasicBlock, [3, 4, 6, 3]),
    '50': (Bottleneck, [3, 4, 6, 3]),
    '101': (Bottleneck, [3, 4, 23, 3]),
    '152': (Bottleneck, [3, 8, 36, 3]),
  }
  return cf_dict[str(depth)]


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, affine, leaky_slope, stride=1):
    super(BasicBlock, self).__init__()
    self.leaky_relu = nn.LeakyReLU(
      negative_slope=leaky_slope, inplace=False)
    self.conv1 = conv3x3(in_planes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes, affine=affine)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes, affine=affine)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
        nn.BatchNorm2d(self.expansion*planes, affine=affine))

  def forward(self, x):
    out = self.leaky_relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = self.leaky_relu(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, affine, leaky_slope, stride=1):
    super(Bottleneck, self).__init__()
    self.leaky_relu = nn.LeakyReLU(
      negative_slope=leaky_slope, inplace=False)
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
    self.bn1 = nn.BatchNorm2d(planes, affine=affine)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
    self.bn2 = nn.BatchNorm2d(planes, affine=affine)
    self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
    self.bn3 = nn.BatchNorm2d(self.expansion*planes, affine=affine)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
        nn.BatchNorm2d(self.expansion*planes, affine=affine))

  def forward(self, x):
    out = self.leaky_relu(self.bn1(self.conv1(x)))
    out = self.leaky_relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    out = self.leaky_relu(out)
    return out


class ResNet(nn.Module):

  def __init__(self, params, num_classes, is_training):
    super(ResNet, self).__init__()

    self.is_training = is_training
    self.num_classes = num_classes
    self.params = params
    self.config = config = self.params.model_params
    self.depth = config['depth']
    self.leaky_slope = config['leaky_slope']
    use_dc_last = getattr(config, 'use_diag_circ', False)
    self.bn_affine = getattr(config, 'bn_affine', True)

    self.in_planes = 16
    block, num_blocks = cfg(self.depth)
    
    self.leaky_relu = nn.LeakyReLU(
      negative_slope=self.leaky_slope, inplace=False)
    self.conv1 = conv3x3(3, 16)
    self.bn1 = nn.BatchNorm2d(16, affine=self.bn_affine)
    self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

    if use_dc_last:
      size = 64*block.expansion
      layers = []
      for i in range(config['n_dc_layers']):
        layers.append(
          DiagonalCirculantLayer(size, size, **self.params.model_params) 
        )
      self.last = nn.Sequential(*layers)
    else:
      self.last = nn.Linear(64*block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, self.bn_affine,
                          self.leaky_slope, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.leaky_relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.avg_pool2d(out, 8)
    out = out.view(out.size(0), -1)
    out = self.last(out)
    return out[:, :self.num_classes]

