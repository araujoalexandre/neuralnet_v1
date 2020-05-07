
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layers import DiagonalCirculantLayer

class DiagonalCirculantModel(nn.Module):

  def __init__(self, params, num_classes, is_training):
    super(DiagonalCirculantModel, self).__init__()
    self.params = params
    n_layers = self.params.model_params['n_layers']
    self.layers = nn.ModuleList([])
    for _ in range(n_layers):
      self.layers.append(
        DiagonalCirculantLayer(3072, 3072, **self.params.model_params))
    self.last = DiagonalCirculantLayer(3072, 10, **self.params.model_params)

  def forward(self, x):
    x = x.view(x.size()[0], -1)
    size = x.size()[-1]
    for layer in self.layers:
      x = layer(x)
      x = F.relu(x)
    return self.last(x)


