
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
        alpha=2., activation='leaky_relu', activation_slope=0.1, **kwargs):
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


