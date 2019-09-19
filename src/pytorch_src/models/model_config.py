
from . import inception_model
from . import lenet_model
from . import resnet_model
from . import vgg_model
from . import wide_resnet_model
from . import circulant_model

_model_name_to_imagenet_model = {
    'trivial': '',
    'vgg11': vgg_model.create_vgg11_model,
    'vgg13': vgg_model.create_vgg13_model,
    'vgg16': vgg_model.create_vgg16_model,
    'vgg19': vgg_model.create_vgg19_model,
    'lenet': lenet_model.LeNet,
    # 'inception4': inception_model.Inceptionv4Model,
    # 'resnet50': '',
    # 'resnet101': '',
    # 'resnet152': '',
}

_model_name_to_cifar_model = {
    # 'trivial': trivial_model.TrivialModel,
    # 'lenet': lenet_model.Lenet5Model,
    # 'resnet20': resnet_model.create_resnet20_cifar_model,
    # 'resnet32': resnet_model.create_resnet32_cifar_model,
    # 'resnet44': resnet_model.create_resnet44_cifar_model,
    # 'resnet56': resnet_model.create_resnet56_cifar_model,
    # 'resnet110': resnet_model.create_resnet110_cifar_model,
    'wide_resnet': wide_resnet_model.WideResnetModel,
    'diagonal_circulant': circulant_model.DiagonalCirculantModel,
}

_model_name_to_mnist_model = {
  # 'trivial': '',
  # 'lenet': lenet_model.Lenet5Model,
}

def _get_model_map(dataset_name):
  """Get name to model map for specified dataset."""
  if dataset_name in ('cifar10', 'cifar100'):
    return _model_name_to_cifar_model
  elif dataset_name == 'mnist':
    return _model_name_to_mnist_model
  elif dataset_name in ('imagenet'):
    return _model_name_to_imagenet_model
  else:
    raise ValueError('Invalid dataset name: {}'.format(dataset_name))


def get_model_config(model_name, dataset_name, params, nclass, is_training):
  """Map model name to model network configuration."""
  model_map = _get_model_map(dataset_name)
  if model_name not in model_map:
    raise ValueError("Invalid model name '{}' for dataset '{}'".format(
                     model_name, dataset_name))
  else:
    return model_map[model_name](params, nclass, is_training)



