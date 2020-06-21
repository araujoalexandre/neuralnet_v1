
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lipschitz_bound.lipschitz_bound import LipschitzBound


class LipschitzRegularization:

  def __init__(self, model, params, reader, local_rank):

    self.params = params
    self.lipschitz_regularization = getattr(
      params, 'lipschitz_regularization', False)

    if self.lipschitz_regularization:

      self.decay = self.params.lipschitz_decay
      self.lipschitz_computation = getattr(
        params, 'lipschitz_computation', 'lipbound') # default lipbound

      if self.lipschitz_computation == 'lipbound':
        if local_rank == 0:
          logging.info("Lipschitz regularization activated with LipBound.")
        self.lip_constants = LipschitzLipBound(model, params, reader)
      elif self.lipschitz_computation == "power_method":
        if local_rank == 0:
          logging.info(
            "Lipschitz regularization activated with Power Iteration {}".format(
              self.params.lipschitz_n_iter))
        self.lip_constants = LipschitzPowerIteration(model, params, reader)
      else:
        raise ValueError("Method for computing Lipschitz not recognized.")

  def get_lip_reg(self, epoch, model):
    if self.lipschitz_regularization and self.decay > 0:
      lip_cst = self.lip_constants.compute(model)
      lip_loss = [torch.log(x) for x in lip_cst]
      return self.decay * sum(lip_loss)
    return 0.



class LipschitzGlobal:

  def __init__(self, model, params):

    self.conv_id = set()
    self.batch_bn_id = set()
    self.diagonal_circulant_id = set()

    for i, module in enumerate(model.modules()):
      name = module.__class__.__name__.lower()
      # logging.info(name)
      # add_it = False
      if 'conv2d' in name:
        # add_it = True
        self.conv_id.add(i)
      elif 'batchnorm' in name:
        # add_it = True
        self.batch_bn_id.add(i)
      elif 'diagonalcirculant' in name:
        # add_it = True
        self.diagonal_circulant_id.add(i)
      # weight = getattr(module, 'weight', False)
      # shape = 0 if weight is False else weight.shape
      # if add_it:
      #   logging.info('{} {} adding it !'.format(name, shape))
      # else:
      #   logging.info('{}'.format(name, shape))

  def _compute_batch_norm(self, module):
    """Compute the Lipschitz of Batch Norm layer."""
    weight = module.weight
    running_var = module.running_var
    eps = module.eps
    lip = torch.max(torch.abs(weight / torch.sqrt(running_var + eps)))
    return lip

  def _compute_diagonal_circulant(self, module):
    """Compute the Lipschitz of Diagonal Circulant layer"""
    lip_circ = torch.max(torch.abs(torch.rfft(module.kernel, 1)))
    lip_diag = torch.max(torch.abs(module.diag))
    return lip_circ, lip_diag

  def compute(self, model):
    """Compute Lipschitz of full Network."""
    lip_loss = []
    for i, module in enumerate(model.modules()):
      name = module.__class__.__name__
      if i in self.conv_id:
        lip_conv = self._compute_conv(i, module)
        lip_loss.append(lip_conv)
      elif i in self.batch_bn_id:
        lip_bn = self._compute_batch_norm(module)
        lip_loss.append(lip_bn)
      elif i in self.diagonal_circulant_id:
        lip_circ, lip_diag = self._compute_diagonal_circulant(module)
        lip_loss.extend([lip_circ, lip_diag])
    return lip_loss



class LipschitzPowerIteration(LipschitzGlobal):

  def __init__(self, model, params, reader):
    super(LipschitzPowerIteration, self).__init__(
      model, params)

    self.model = model
    self.n_iter = params.lipschitz_n_iter

    self.u = {}

    input_size = reader.batch_shape[1:]
    input_size = (1,) + input_size

    # record the input size for each module
    self.model_input_size = {}
    def wrapper_function(module_id):
      def store_input_sizes(module, input, output):
        self.model_input_size[module_id] = list(input[0].shape)
      return store_input_sizes
    self.execute_through_model(wrapper_function, input_size)

  def execute_through_model(self, function, input_size):
    """ Execute `function` through the model"""
    handles = []
    for module_id, module in enumerate(self.model.modules()):
      handle = module.register_forward_hook(function(module_id))
      handles.append(handle)

    x = torch.zeros(*input_size)
    x = x.cuda()
    self.model(x)

    # Remove hooks
    for handle in handles:
      handle.remove()

  def _compute_conv(self, i, module):
    """Compute a bound on the Lipchitz of Convolution layer."""

    kernel = module.weight.clone()
    padding = module.padding[0]
    shape = list(self.model_input_size[i])
    
    pad = (padding, padding, padding, padding)
    pad_ = (-padding, -padding, -padding, -padding)

    def normalize(arr):
      norm = torch.sqrt((arr ** 2).sum())
      return arr / (norm + 1e-12)
      
    def power_iteration_conv(u, w_mat, n_iter):
      u_ = u
      for i in range(n_iter):
        v_ = normalize(F.conv2d(F.pad(u_, pad), kernel))
        u_ = normalize(F.pad(F.conv_transpose2d(v_, kernel), pad_))
      return u_, v_
   
    kshape = str(shape)
    if kshape not in self.u.keys():
      self.u[kshape] = torch.normal(0, 0.05, size=shape, device='cuda')
    u_hat, v_hat = power_iteration_conv(self.u[kshape].clone(), kernel, self.n_iter)
    
    z = F.conv2d(F.pad(u_hat, pad), kernel)
    sigma = torch.max(torch.mul(z, v_hat).sum())
    return sigma



class LipschitzLipBound(LipschitzGlobal):

  def __init__(self, model, params, *args):
    super(LipschitzLipBound, self).__init__(
      model, params)

    self.lip_bound_cls = {}
    self.sample = params.lipschitz_bound_sample
    
    # # we pre-create LipschitzBound for each kernel
    # for i, module in enumerate(model.modules()):
    #   name = module.__class__.__name__
    #   print(name)
    #   if name == 'Conv2d':
    #     padding = module.padding[0]
    #     kernel = module.weight
    #     self.lip_bound_cls[i] = \
    #       LipschitzBound(kernel.shape, padding, sample=self.sample)

    # we pre-create LipschitzBound for each kernel
    for i, module in enumerate(model.modules()):
      name = module.__class__.__name__
      if len(getattr(module, 'weight', [])):
        if len(module.weight.shape) == 4:
          padding = module.padding[0]
          kernel = module.weight
          # logging.info('{} {}'.format(
          #   module.__class__.__name__, len(kernel.shape)))
          self.lip_bound_cls[i] = \
            LipschitzBound(kernel.shape, padding, sample=self.sample)

  def _compute_conv(self, i, module):
    """Compute a bound on the Lipchitz of Convolution layer."""
    padding = module.padding[0]
    kernel = module.weight
    lip = self.lip_bound_cls[i].compute(kernel)
    return lip



