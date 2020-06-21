import logging
import copy
import multiprocessing

import torch
from torch.distributions import normal, laplace, uniform, bernoulli
from advertorch import attacks

def set_default_param_values_and_env_vars(params):
  """Sets up the default param values and environment variables ."""
  # Sets GPU thread settings
  if params.device == 'gpu':
    if params.gpu_thread_mode not in ['global', 'gpu_shared', 'gpu_private']:
      raise ValueError('Invalid gpu_thread_mode: %s' % params.gpu_thread_mode)

    if params.per_gpu_thread_count and params.gpu_thread_mode == 'global':
      raise ValueError(
          'Invalid per_gpu_thread_count with gpu_thread_mode=global: %s' %
          params.per_gpu_thread_count)
    # Default to two threads. One for the device compute and the other for
    # memory copies.
    per_gpu_thread_count = params.per_gpu_thread_count or 2
    total_gpu_thread_count = per_gpu_thread_count * params.num_gpus

    cpu_count = multiprocessing.cpu_count()
    if not params.num_inter_threads and params.gpu_thread_mode in [
        'gpu_private', 'gpu_shared'
    ]:
      main_thread_count = max(cpu_count - total_gpu_thread_count, 1)
      params.num_inter_threads = main_thread_count

    # From the total cpu thread count, subtract the total_gpu_thread_count,
    # and then 2 threads per GPU device for event monitoring and sending /
    # receiving tensors
    num_monitoring_threads = 2 * params.num_gpus
    num_private_threads = max(
        cpu_count - total_gpu_thread_count - num_monitoring_threads, 1)
    if params.datasets_num_private_threads == 0:
      params.datasets_num_private_threads = num_private_threads
  return params


def get_attack(model, num_classes, attack_name, attack_params):
  if attack_name == 'carlini':
    attack = attacks.CarliniWagnerL2Attack(model, num_classes, **attack_params)
  elif attack_name == 'elasticnet':
    attack = attacks.ElasticNetL1Attack(model, num_classes, **attack_params)
  elif attack_name == 'pgd':
    norm = attack_params['norm']
    del attack_params['norm']
    if norm == 'inf':
      attack = attacks.LinfPGDAttack(model, **attack_params)
    elif norm == 'l1':
      attack = attacks.SparseL1PGDAttack(model, **attack_params)
    elif norm == 'l2':
      attack = attacks.L2PGDAttack(model, **attack_params)
    else:
      raise ValueError("Norm not recognized for PGD attack.")
  elif attack_name == 'fgsm':
    attack = GradientSignAttack(model, **attack_params)
  else:
    raise ValueError("Attack name not recognized for adv training.")
  return attack





class MixtureNoise:

  def __init__(self, noise1, noise2, method, prob=0.5):
    self.noise1 = noise1
    self.noise2 = noise2
    self.bernoulli = bernoulli.Bernoulli(probs=prob)

    if method == 'rand':
      self.sample = self.sample_rand
    elif method == 'sum':
      self.sample = self.sample_sum

  def sample_rand(self, shape):
    if self.bernoulli.sample():
      return self.noise1.sample(shape)
    else:
      return self.noise2.sample(shape)

  def sample_sum(self, shape):
    return self.noise1.sample(shape) + self.noise2.sample(shape)



class Uniform:

  def __init__(self, low, high, scale):
    self.scale = scale
    self.noise = uniform.Uniform(low, high)

  def sample(self, shape):
    return ((self.noise.sample(shape) * 2) - 1) * self.scale


class AddNoise:

  def __init__(self, params):

    self.params = params
    dist = self.params.noise['distribution']
    loc_normal = self.params.noise['loc_normal']
    scale_normal = self.params.noise['scale_normal']
    scale_uniform = self.params.noise['scale_uniform']
    low = self.params.noise['low']
    high = self.params.noise['high']

    if dist == "normal":
      self.noise = normal.Normal(loc_normal, scale_normal)
    elif dist == 'uniform':
      self.noise = Uniform(low, high, scale_uniform)
    elif dist == 'uniform+normal':
      noise1 = normal.Normal(loc_normal, scale_normal)
      noise2 = Uniform(low, high, scale_uniform)
      self.noise = MixtureNoise(noise1, noise2, method='sum')
    elif dist == 'mix_rand_uniform_normal':
      noise1 = normal.Normal(loc_normal, scale_normal)
      noise2 = Uniform(low, high, scale_uniform)
      self.noise = MixtureNoise(noise1, noise2, method='rand')
    else:
      raise ValueError('Noise not recognized.')
    logging.info('Noise Injection {}'.format(dist))

  def __call__(self, x):
    if x.is_cuda:
      return x + self.noise.sample(x.shape).cuda()
    return x + self.noise.sample(x.shape)


class EOTWrapper(torch.nn.Module):

  def __init__(self, model, num_classes, params):
    super(EOTWrapper, self).__init__()
    self.model = model
    self.model.eval()
    self.num_classes = num_classes

    self.eot_samples = params.eot_samples
    logging.info('Using EOT Samples {}'.format(self.eot_samples))
    self.noise = AddNoise(params)

  def forward(self, x):
    bs = x.shape[0]
    x = torch.repeat_interleave(x, repeats=self.eot_samples, dim=0)
    y = self.model(self.noise(x))
    y = y.view(bs, self.eot_samples, self.num_classes)
    return torch.mean(y, dim=1)



class CrossEntropyLabelSmooth(torch.nn.Module):
  def __init__(self, num_classes, epsilon, reduction='mean'):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.reduction = reduction
    self.logsoftmax = torch.nn.LogSoftmax(dim=1)

  def forward(self, input, target):  # pylint: disable=redefined-builtin
    log_probs = self.logsoftmax(input)
    targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
    if self.epsilon > 0.0:
      targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    targets = targets.detach()
    loss = (-targets * log_probs)

    if self.reduction in ['avg', 'mean']:
      loss = torch.mean(torch.sum(loss, dim=1))
    elif self.reduction == 'sum':
      loss = loss.sum()
    return loss


class EMA:

  def __init__(self, mu):
    self.mu = mu
    self.shadow = {}

  def state_dict(self):
    return copy.deepcopy(self.shadow)

  def __len__(self):
    return len(self.shadow)

  def __call__(self, module, step=None):
    if step is None:
      mu = self.mu
    else:
      # see: tensorflow doc ExponentialMovingAverage
      mu = min(self.mu, (1. + step) / (10 + step))
    for name, x in module.state_dict().items():
      if name in self.shadow:
        new_average = (1.0 - mu) * x + mu * self.shadow[name]
        self.shadow[name] = new_average.clone()
      else:
        self.shadow[name] = x.clone()
