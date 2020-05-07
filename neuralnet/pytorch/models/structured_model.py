
from inspect import signature
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .structure import layer

import kymatio
from kymatio import Scattering2D

class LDRModel(nn.Module):

  def __init__(self, params, num_classes, is_training):
    super(LDRModel, self).__init__()
    self.params = params
    self.model_params = params.model_params
    rank = self.model_params['rank']
    class_type = self.model_params['class_type']

    shape = 3 * 81 * 8 * 8
    self.scattering = Scattering2D(J=2, shape=(32, 32))
    self.scattering = self.scattering.cuda()
    self.batch_norm = nn.BatchNorm1d(shape)

    if class_type == 'ldr-sd':
      self.ldr = layer.LDRSubdiagonal(layer_size=shape, corner=False, r=rank)
    elif class_type == 'ldr-td':
      self.ldr = layer.LDRTridiagonal(layer_size=shape, corner=False, r=rank)
    elif class_type == 'toeplitz':
      self.ldr  = layer.ToeplitzLike(layer_size=shape, corner=False, r=rank)
    self.ldr = self.ldr.cuda()

  def forward(self, x):
    x = self.scattering(x)
    x = x.reshape(x.size()[0], -1)
    x = self.batch_norm(x)
    x = F.relu(x)
    x = self.ldr(x)[..., :10]
    return x


class LDRMultiLayerModel(nn.Module):

  def __init__(self, params, num_classes, is_training):
    super(LDRMultiLayerModel, self).__init__()
    self.params = params
    self.model_params = params.model_params
    self.n_layers = self.params.model_params['n_layers']
    rank = self.model_params['rank']
    class_type = self.model_params['class_type']

    shape = 3 * 81 * 8 * 8
    self.scattering = Scattering2D(J=2, shape=(32, 32))
    self.scattering = self.scattering.cuda()
    self.batch_norm = nn.BatchNorm1d(shape)

    self.layers = nn.ModuleList([])
    if class_type == 'ldr-sd':
      for _ in range(self.params.model_params['n_layers']):
        self.layers.append(
          layer.LDRSubdiagonal(layer_size=shape, corner=False, r=rank))
      self.last = layer.LDRSubdiagonal(layer_size=shape, corner=False, r=rank)

    elif class_type == 'ldr-td':
      for _ in range(self.params.model_params['n_layers']):
        l = layer.LDRTridiagonal(layer_size=shape, corner=False, r=rank)
        self.layers.append(l)
      self.last = layer.LDRTridiagonal(layer_size=shape, corner=False, r=rank)

    elif class_type == 'toeplitz':
      self.ldr  = layer.ToeplitzLike(layer_size=shape, corner=False, r=rank)
      for _ in range(self.params.model_params['n_layers']):
        l = layer.LDRTridiagonal(layer_size=shape, corner=False, r=rank)
        self.layers.append(l)
      self.last = layer.LDRTridiagonal(layer_size=shape, corner=False, r=rank)


  def forward(self, x):
    x = self.scattering(x)
    x = x.view(x.size()[0], -1)
    x = self.batch_norm(x)
    x = F.relu(x)
    for layer in self.layers:
      x = layer(x)
      x = F.relu(x)
    return self.last(x)[..., :10]




class WLDRFC:
  """
  LDR layer (single weight matrix), followed by FC and softmax
  """
  def name(self):
    return self.W.name()+'u'

  def args(class_type='unconstrained', layer_size=-1, r=1, fc_size = 512): pass
  def reset_parameters(self):
    if self.layer_size == -1: self.layer_size = self.in_size
    self.W = layer.StructuredLinear(self.class_type, layer_size=self.layer_size, r=self.r)
    self.fc = nn.Linear(3*1024, self.fc_size)
    self.logits = nn.Linear(self.fc_size, 10)

  def forward(self, x):
    x = self.W(x)
    x = F.relu(self.fc(x))
    x = self.logits(x)
    return x


class LDRFC:
  """
  LDR layer with channels, followed by FC and softmax
  """
  def __init__(class_type='t', r=1, channels=3, fc_size=512):

    self.class_type = class_type
    self.r = r
    self.channels=channels
    self.n = 1024
    self.LDR1 = ldr.LDR(self.class_type, 3, self.channels, self.r, self.n, bias=True)
    self.fc = nn.Linear(self.channels*self.n, self.fc_size)
    self.logits = nn.Linear(self.fc_size, 10)

  def forward(self, x):
    x = x.view(-1, 3, 1024)
    x = x.transpose(0,1).contiguous().view(3, -1, self.n)
    x = F.relu(self.LDR1(x))
    x = x.transpose(0, 1) # swap batches and channels axis
    x = x.contiguous().view(-1, self.channels*self.n)
    x = F.relu(self.fc(x))
    x = self.logits(x)
    return x



class LDRLDR:
  """
  LDR layer (either 3 channels or one wide matrix), followed by another LDR layer, then softmax
  intended for 3-channel images of size 1024 (e.g. CIFAR-10)
  """
  def name(self):
    # w = 'wide' if not self.channels else ''
    return self.LDR1.name() + self.LDR211.name()

  def args(class1='toeplitz', class2='toeplitz', channels=False, rank1=48, rank2=16): pass
  def reset_parameters(self):
    self.n = 1024
    self.fc_size = self.n // 2

    if self.channels:
        self.LDR1 = ldr.LDR(self.class1, 3, 3, self.rank1, self.n)
    else:
        self.LDR1 = layer.StructuredLinear(self.class1, layer_size=3*self.n, r=self.rank1)

    self.LDR211 = layer.StructuredLinear(self.class2, layer_size=self.fc_size, r=self.rank2)
    self.LDR212 = layer.StructuredLinear(self.class2, layer_size=self.fc_size, r=self.rank2)
    self.LDR221 = layer.StructuredLinear(self.class2, layer_size=self.fc_size, r=self.rank2)
    self.LDR222 = layer.StructuredLinear(self.class2, layer_size=self.fc_size, r=self.rank2)
    self.LDR231 = layer.StructuredLinear(self.class2, layer_size=self.fc_size, r=self.rank2)
    self.LDR232 = layer.StructuredLinear(self.class2, layer_size=self.fc_size, r=self.rank2)
    self.b = Parameter(torch.zeros(self.fc_size))
    self.logits = nn.Linear(self.fc_size, 10)

  def forward(self, x):
    if self.channels:
        x = x.view(-1, 3, self.n)
        x = x.transpose(0,1).contiguous().view(3, -1, self.n)
        x = F.relu(self.LDR1(x))
    else:
        x = F.relu(self.LDR1(x))
        x = x.view(-1, 3, self.n)
        x = x.transpose(0,1).contiguous().view(3, -1, self.n)
    x11 = x[0][:,:self.fc_size]
    x12 = x[0][:,self.fc_size:]
    x21 = x[1][:,:self.fc_size]
    x22 = x[1][:,self.fc_size:]
    x31 = x[2][:,:self.fc_size]
    x32 = x[2][:,self.fc_size:]
    x = F.relu(
      self.LDR211(x11) + self.LDR212(x12) + \
      self.LDR221(x21) + self.LDR222(x22) + \
      self.LDR231(x31) + self.LDR232(x32) + self.b)
    x = self.logits(x)
    return x




