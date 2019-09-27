
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import kymatio
from kymatio import Scattering2D

from .layers import DiagonalCirculantLayer

class ScatteringCirculantModel(nn.Module):

  def __init__(self, params, num_classes, is_training):
    super(ScatteringCirculantModel, self).__init__()
    self.params = params
    self.model_params = params.model_params
    self.use_batch_norm = self.model_params.get('use_batch_norm', False)
    n_layers = self.model_params['n_layers']

    shape = 3 * 81 * 8 * 8
    self.scattering = Scattering2D(J=2, shape=(32, 32))
    self.scattering = self.scattering.cuda()
    if self.use_batch_norm:
      self.batch_norm = nn.BatchNorm1d(shape)

    self.layers = nn.ModuleList([])
    for _ in range(self.params.model_params['n_layers']):
      self.layers.append(
        DiagonalCirculantLayer(shape, shape, **self.params.model_params))
    self.last = DiagonalCirculantLayer(shape, 10, **self.params.model_params)

  def forward(self, x):
    x = self.scattering(x)
    x = x.view(x.size()[0], -1)
    if self.use_batch_norm:
      x = self.batch_norm(x)
    x = F.relu(x)
    for layer in self.layers:
      x = layer(x)
      x = F.relu(x)
    return self.last(x)



class ScatteringPoolingCirculantModel(nn.Module):

  def __init__(self, params, num_classes, is_training):
    super(ScatteringPoolingCirculantModel, self).__init__()
    self.params = params
    self.model_params = params.model_params
    self.use_batch_norm = self.model_params.get('use_batch_norm', False)
    n_layers = self.model_params['n_layers']
    pooling = self.model_params.get('pooling', None)
    assert pooling is not None, "Pooling not defined in config"

    shape = 3 * 81 * 8 * 8
    self.scattering = Scattering2D(J=2, shape=(32, 32))
    self.scattering = self.scattering.cuda()
    if pooling == 'avg':
      self.pooling = nn.AvgPool2d(2)
    elif pooling == 'max':
      self.pooling = nn.MaxPool2d(2)

    shape = 3 * 81 * 4 * 4
    if self.use_batch_norm:
      self.batch_norm = nn.BatchNorm1d(shape)

    self.layers = nn.ModuleList([])
    for _ in range(self.params.model_params['n_layers']):
      self.layers.append(
        DiagonalCirculantLayer(shape, shape, **self.params.model_params))
    self.last = DiagonalCirculantLayer(shape, 10, **self.params.model_params)

  def forward(self, x):
    x = self.scattering(x)
    x = x.reshape(-1, 81, 8, 8)
    x = self.pooling(x)
    x = x.reshape(-1, 3, 81, 4, 4)
    x = x.view(x.size()[0], -1)
    if self.use_batch_norm:
      x = self.batch_norm(x)
    x = F.relu(x)
    for layer in self.layers:
      x = layer(x)
      x = F.relu(x)
    return self.last(x)



class ScatteringByChannelCirculantModel(nn.Module):

  def __init__(self, params, num_classes, is_training):
    super(ScatteringByChannelCirculantModel, self).__init__()
    self.params = params
    self.model_params = params.model_params
    self.use_batch_norm = self.model_params.get('use_batch_norm', False)
    n_layers = self.model_params['n_layers']
    pooling = self.model_params.get('pooling', None)
    assert pooling is not None, "Pooling not defined in config"

    shape = 81 * 8 * 8
    self.scattering = Scattering2D(J=2, shape=(32, 32))
    self.scattering = self.scattering.cuda()
    if self.use_batch_norm:
      self.batch_norm1 = nn.BatchNorm1d(shape)
      self.batch_norm2 = nn.BatchNorm1d(shape)
      self.batch_norm3 = nn.BatchNorm1d(shape)


    self.circ_channel1 = DiagonalCirculantLayer(shape, shape, **self.params.model_params)
    self.circ_channel2 = DiagonalCirculantLayer(shape, shape, **self.params.model_params)
    self.circ_channel3 = DiagonalCirculantLayer(shape, shape, **self.params.model_params)

    self.layers = nn.ModuleList([])
    for _ in range(self.params.model_params['n_layers']):
      self.layers.append(
        DiagonalCirculantLayer(shape, shape, **self.params.model_params))
      self.last = DiagonalCirculantLayer(shape, 10, **self.params.model_params)

  def forward(self, x):
    x1, x2, x3 = torch.unbind(x, dim=1)
    x1 = self.scattering(x1.contiguous())
    x2 = self.scattering(x2.contiguous())
    x3 = self.scattering(x3.contiguous())

    x1 = x1.view(x1.size()[0], -1)
    x2 = x2.view(x1.size()[0], -1)
    x3 = x3.view(x3.size()[0], -1)

    if self.use_batch_norm:
      x1 = self.batch_norm1(x1)
      x2 = self.batch_norm2(x2)
      x3 = self.batch_norm3(x3)

    x1 = F.relu(x1)
    x2 = F.relu(x2)
    x3 = F.relu(x3)

    x1 = self.circ_channel1(x1)
    x2 = self.circ_channel2(x2)
    x3 = self.circ_channel3(x3)
    x = x1 + x2 + x3
    x = F.relu(x)

    for layer in self.layers:
      x = layer(x)
      x = F.relu(x)
    return self.last(x)






