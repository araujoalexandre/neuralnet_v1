
import logging
import numpy as np
import torch
from lipschitz_bound.lipschitz_bound import LipschitzBound


class LipschitzRegularization:

  def __init__(self, model, params):

    self.lipschitz_regularization = getattr(
      params, 'lipschitz_regularization', False)
    self.lipschitz_normalization = getattr(
      params, 'lipschitz_normalization', False)
    self.decay = getattr(params, 'lipschitz_decay', False)
    self.method = getattr(params, 'lipschitz_method', False)
    
    if self.lipschitz_regularization:
      logging.warning("Lipschitz regularization activated.")
      self.lip_constants = LipschitzConstants(model, params)

  def get_lip_reg(self, epoch, model):
    if self.lipschitz_regularization:
      self.lip_constants.clear()
      lip_cst = self.lip_constants.compute_lipschitz_cst(model)
      if self.method == 'sum':
        lip_loss = sum(lip_cst)
        return self.decay * lip_loss
      elif self.method == 'product':
        lip_loss = [torch.log(x + 1) for x in lip_cst]
        return self.decay * sum(lip_loss)
      else:
        raise "Lipschitz regularization method not recognized."
    return 0.





class LipschitzConstants:

  def __init__(self, model, params):

    self.lip_loss = []
    self.lip_bound_cls = {}
    self.sample = params.lipschitz_bound_sample

    self.conv_id = set()
    self.batch_bn_id = set()
    self.diagonal_circulant_id = set()

    # we pre-create LipschitzBound for each kernel
    for i, module in enumerate(model.modules()):
      name = module.__class__.__name__
      if name == 'Conv2d':
        padding = module.padding[0]
        kernel = module.weight
        self.lip_bound_cls[i] = \
          LipschitzBound(kernel.shape, padding, sample=self.sample)
        self.conv_id.add(i)
      elif name == 'BatchNorm2d':
        self.batch_bn_id.add(i)
      elif name == 'DiagonalCirculantLayer':
        self.diagonal_circulant_id.add(i)

  def clear(self):
    self.lip_loss = []

  def _compute_conv(self, i, module):
    """Compute a bound on the Lipchitz of Convolution layer."""
    padding = module.padding[0]
    kernel = module.weight
    # self.lip_bound_cls[i].repackage()
    lip = self.lip_bound_cls[i].compute(kernel)
    return lip

  def _compute_batch_norm(self, module):
    """Compute the Lipschitz of Batch Norm layer."""
    weight = module.weight
    running_var = module.running_var
    eps = module.eps
    lip = torch.max(torch.abs(weight / torch.sqrt(running_var + eps)))
    return lip

  def _compute_diagonal_circulant(self, module):
    lip_circ = torch.max(torch.abs(torch.rfft(module.kernel, 1)))
    lip_diag = torch.max(torch.abs(module.diag))
    return lip_circ, lip_diag

  def compute_lipschitz_cst(self, model):
    """Compute Lipschitz of full Network."""
    for i, module in enumerate(model.modules()):
      name = module.__class__.__name__
      if i in self.conv_id:
        lip_conv = self._compute_conv(i, module)
        self.lip_loss.append(lip_conv)
      elif i in self.batch_bn_id:
        lip_bn = self._compute_batch_norm(module)
        self.lip_loss.append(lip_bn)
      elif i in self.diagonal_circulant_id:
        lip_circ, lip_diag = self._compute_diagonal_circulant(module)
        self.lip_loss.extend([lip_circ, lip_diag])
    return self.lip_loss



class LipschitzNormalization:

  def _normalize_conv(self, module):
    """Normalize Convolutional kernel with Lipschitz."""
    padding = module.padding[0]
    kernel = module.weight
    lb = LipschitzBound(kernel, padding, sample=50)
    lip_conv = lb.compute()
    if lip_conv > 1:
      module.weight.data = module.weight.data / lip_conv

  def _normalize_diagonal_circulant(self, module):
    """Normalize Diagonal Circulant layer with Lipschitz."""
    lip_circ = torch.max(torch.abs(torch.rfft(module.kernel, 1)))
    lip_diag = torch.max(torch.abs(module.diag))
    if lip_circ > 1:
      module.kernel.data = module.kernel.data / lip_circ
    if lip_diag > 1:
      module.diag.data = module.diag.data / lip_diag

  def normalize_network(self, model):
    """Normalize each layer of network with Lipschitz"""
    for module in self.model.modules():
      name = module.__class__.__name__
      if name == 'Conv2d':
        self._normalize_conv(module)
      if name == 'DiagonalCirculantLayer':
        self._normalize_diagonal_circulant(module)



    # else: # no reg, but we compute the lip anyway
    #
    #   for module in self.model.modules():
    #     if 'conv' not in module.__class__.__name__.lower():
    #       continue
    #     n_layers += 1
    #     padding = module.padding[0]
    #     kernel = module.weight.data
    #     lb = LipschitzBound(kernel, padding, sample=50)
    #     lip_bound = lb.compute()
    #     lip_loss += lip_bound
    #     lip_log_loss = torch.log(lip_bound + 1)
    #   lip_mean = lip_loss / n_layers
    #
    # # if self.lip_margin:   
    # #   y_onehot = torch.zeros((len(labels), 10))
    # #   y_onehot = y_onehot.scatter_(1, y, 1)
    # #   y_onehot[y_onehot == 1] = -1
    # #   y_onehot[y_onehot == 0] = 1
    # #   y_onehot[y_onehot == -1] = 0
    # #   
    # #   mask = torch.max(outputs, axis=1)[1] == y.reshape(-1)
    # #   outputs[mask, :] += torch.sign(outputs[mask, :]) * y_onehot[mask, :] *
    # #   torch.sqrt(2) * self.params.lip_margin_cst * lip_loss
    #
    # loss = self.criterion(outputs, labels.cuda())
    #
    #


