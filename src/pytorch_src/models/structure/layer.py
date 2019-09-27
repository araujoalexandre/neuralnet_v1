
import numpy as np
import torch

import torch.nn as nn
from torch.nn.parameter import Parameter

from . import toeplitz as toep
from . import krylov as kry
from . import circulant as circ
from . import fastfood as ff


class Layer(nn.Module):

  def __init__(self, layer_size=None, bias=True, r=1, **kwargs):
    super(Layer, self).__init__()
    self.layer_size = layer_size
    assert self.layer_size is not None
    self.bias = bias
    self.b = None
    if self.bias:
      self.b = Parameter(torch.zeros(self.layer_size))

    self.r = r
    self.G = Parameter(torch.Tensor(self.r, self.layer_size))
    self.H = Parameter(torch.Tensor(self.r, self.layer_size))
    self.init_stddev = np.power(1. / (self.r * self.layer_size), 1/2)
    torch.nn.init.normal_(self.G, std=self.init_stddev)
    torch.nn.init.normal_(self.H, std=self.init_stddev)

  def apply_bias(self, out):
    if self.b is not None:
      return self.b + out
    else:
      return out


class ToeplitzLike(Layer):

  def __init__(self, corner=True, **kwargs):
    super(ToeplitzLike, self).__init__(**kwargs)
    self.corner = corner

  def forward(self, x):
    out = toep.toeplitz_mult(self.G, self.H, x, self.corner)
    return self.apply_bias(out)


class LDRSubdiagonal(Layer):

  def __init__(self, tie_operators=False, corner=False, **kwargs):
    super(LDRSubdiagonal, self).__init__(**kwargs)

    self.tie_operators = tie_operators
    self.corner = corner

    self.subd_A = Parameter(torch.ones(self.layer_size-1))
    if self.tie_operators:
      self.subd_B = self.subd_A
    else:
      self.subd_B = Parameter(torch.ones(self.layer_size-1))

    if corner:
      self.corner_A = Parameter(torch.tensor(0.0))
      self.corner_B = Parameter(torch.tensor(0.0))

  def forward(self, x):
    if not self.corner:
      out = kry.subdiag_mult(self.subd_A, self.subd_B, self.G, self.H, x)
      return self.apply_bias(out)
    out = kry.subdiag_mult_cuda(
      self.subd_A, self.subd_B, self.G, self.H, x,
      corner_A=self.corner_A, corner_B=self.corner_B)
    return self.apply_bias(out)



class LDRTridiagonal(Layer):

  def __init__(self, tie_operators=False, corner=False, **kwargs):
    super(LDRTridiagonal, self).__init__(**kwargs)

    self.tie_operators = tie_operators
    self.corner = corner

    self.subd_A = Parameter(torch.ones(self.layer_size-1))
    self.diag_A = Parameter(torch.zeros(self.layer_size))
    self.supd_A = Parameter(torch.zeros(self.layer_size-1))
    if self.tie_operators:
      self.subd_B = self.subd_A
      self.diag_B = self.diag_A
      self.supd_B = self.supd_A
    else:
      self.subd_B = Parameter(torch.ones(self.layer_size-1))
      self.diag_B = Parameter(torch.zeros(self.layer_size))
      self.supd_B = Parameter(torch.zeros(self.layer_size-1))

    if not self.corner:
      self.corners_A = (0.0, 0.0)
      self.corners_B = (0.0, 0.0)
    else:
      self.corners_A = (Parameter(torch.tensor(0.0)), Parameter(torch.tensor(0.0)))
      self.corners_B = (Parameter(torch.tensor(0.0)), Parameter(torch.tensor(0.0)))

  def forward(self, x):
    out = kry.tridiag_mult_slow(
      self.subd_A, self.diag_A, self.supd_A,
      self.subd_B, self.diag_B, self.supd_B,
      self.G, self.H, x, corners_A=self.corners_A, corners_B=self.corners_B)
    return self.apply_bias(out)





