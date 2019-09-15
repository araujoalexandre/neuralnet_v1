
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1, bias=True):
  return nn.Conv2d(
    in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    init.xavier_uniform_(m.weight, gain=np.sqrt(2))
    init.constant_(m.bias, 0)
  elif classname.find('BatchNorm') != -1:
    init.constant_(m.weight, 1)
    init.constant_(m.bias, 0)


class wide_basic(nn.Module):
  def __init__(self, in_planes, planes, dropout_rate, stride=1, bias=True):
    super(wide_basic, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv1 = nn.Conv2d(
      in_planes, planes, kernel_size=3, padding=1, bias=bias)
    self.dropout = nn.Dropout(p=dropout_rate)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(
      planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != planes:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=bias))

  def forward(self, x):
    out = self.dropout(self.conv1(F.relu(self.bn1(x))))
    out = self.conv2(F.relu(self.bn2(out)))
    out += self.shortcut(x)

    return out


class WideResnetModel(nn.Module):
  """Wide ResNet model.
     https://arxiv.org/abs/1605.07146
  """
  def __init__(self, params, num_classes, is_training):
    super(WideResnetModel, self).__init__()
    self.in_planes = 16

    self.is_training = is_training
    self.params = params
    self.config = config = self.params.model_params

    k = config['widen_factor']
    depth = config['depth']
    dropout_rate = config['dropout']
    bias = config.get('bias', None) or True

    assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
    n = (depth-4)/6

    nStages = [16, 16*k, 32*k, 64*k]

    self.conv1 = conv3x3(3, nStages[0])
    self.layer1 = self._wide_layer(
      wide_basic, nStages[1], n, dropout_rate, stride=1, bias=bias)
    self.layer2 = self._wide_layer(
      wide_basic, nStages[2], n, dropout_rate, stride=2, bias=bias)
    self.layer3 = self._wide_layer(
      wide_basic, nStages[3], n, dropout_rate, stride=2, bias=bias)
    self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)

    self.linear = nn.Linear(nStages[3], num_classes)

  def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, bias=False):
    strides = [stride] + [1]*int(num_blocks-1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, dropout_rate, stride, bias=bias))
      self.in_planes = planes
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.conv1(x)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.relu(self.bn1(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out
