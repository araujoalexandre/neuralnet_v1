
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



def complex_mult(X, Y):
  assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
  return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)

def circulant_multiply(k, x):
  """ Multiply circulant matrix with first column k by x
  Parameters:
    k: (n, )
    x: (batch_size, n) or (n, )
  Return:
    prod: (batch_size, n) or (n, )
  """
  k_fft = torch.rfft(k, 1)
  x_fft = torch.rfft(x, 1)
  mul = complex_mult(k_fft, x_fft)
  return torch.irfft(mul, 1, signal_sizes=(k.shape[-1], ))


class DiagonalCirculantLayer(nn.Module):

  def __init__(self, shape_in, shape_out, use_diag=True, use_bias=True,
        alpha=2., **kwargs):
    super(DiagonalCirculantLayer, self).__init__()
    self.use_diag, self.use_bias = use_diag, use_bias
    self.shape_in, self.shape_out = shape_in, shape_out
    size = np.max([shape_in, shape_out])

    self.size = np.max([shape_in, shape_out])
    shape = (self.size, )

    self.kernel = nn.Parameter(torch.Tensor(shape_in))
    nn.init.normal_(self.kernel, std=np.sqrt(alpha / shape_in))

    self.padding = True
    if shape_out < shape_in:
        self.padding = False

    if use_diag:
      diag = np.float32(np.random.choice([-1, 1], size=(shape_out, )))
      self.diag = nn.Parameter(torch.Tensor(diag))

    if use_bias:
      self.bias = nn.Parameter(torch.Tensor(shape_out))
      nn.init.constant_(self.bias, 0.1)

  def normalize(self):
    max_circ = torch.max(torch.abs(torch.rfft(self.kernel, 1)))
    if max_circ > 1:
      self.kernel.data = self.kernel.data / max_circ
    max_diag = torch.max(torch.abs(self.diag))
    if max_diag > 1:
      self.diag.data = self.diag.data / max_diag

  def get_sv_max(self):
    max_circ = torch.max(torch.abs(torch.rfft(self.kernel, 1)))
    max_diag = torch.max(torch.abs(self.diag))
    return max_circ, max_diag

  def forward(self, x):
    padding_size = np.abs(self.size - self.shape_in)
    paddings = (0, padding_size)
    if self.padding:
      pad_layer = nn.ConstantPad1d(paddings, 0)
      x = pad_layer(x)
    x = circulant_multiply(self.kernel, x)
    x = x[..., :self.shape_out]
    if self.use_diag:
      x = torch.mul(x, self.diag)
    if self.use_bias:
      x = x + self.bias
    return x



