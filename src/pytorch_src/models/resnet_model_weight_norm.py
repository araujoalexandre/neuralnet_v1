import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys

from .layers import DiagonalCirculantLayer

def conv3x3(in_planes, out_planes, stride=1):
  conv = nn.Conv2d(
    in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
  return nn.utils.weight_norm(conv)


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

  def __init__(self, in_planes, planes, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(in_planes, planes, stride)
    self.conv1 = nn.utils.weight_norm(self.conv1)
    self.conv2 = conv3x3(planes, planes)
    self.conv2 = nn.utils.weight_norm(self.conv2)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
        nn.utils.weight_norm(nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                  stride=stride, bias=True)))

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = self.conv2(out)
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
    self.conv1 = nn.utils.weight_norm(self.conv1)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
    self.conv2 = nn.utils.weight_norm(self.conv2)
    self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
    self.conv3 = nn.utils.weight_norm(self.conv3)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion*planes:
      self.shortcut = nn.Sequential(
        nn.utils.weight_norm(nn.Conv2d(in_planes, self.expansion*planes,
                                       kernel_size=1, stride=stride,
                                       bias=True)))

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out))
    out = self.conv3(out)
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNet(nn.Module):

  def __init__(self, params, num_classes, is_training):
    super(ResNet, self).__init__()

    self.is_training = is_training
    self.num_classes = num_classes
    self.params = params
    self.config = config = self.params.model_params

    self.in_planes = 16
    block, num_blocks = cfg(config['depth'])

    self.conv1 = conv3x3(3, 16)
    self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

    if config['use_diag_circ']:
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
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.avg_pool2d(out, 8)
    out = out.view(out.size(0), -1)
    out = self.last(out)
    return out[:, :self.num_classes]

