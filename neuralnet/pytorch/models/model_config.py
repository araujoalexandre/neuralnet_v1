
from . import inception_model
from . import resnet_model_cifar
from . import resnet_model_imagenet
from . import vgg_model
from . import wide_resnet_model
from . import circulant_model
from . import scattering_model
from . import structured_model
from .efficientnet.model import create_efficientnet_model

_model_name_to_imagenet_model = {
    'vgg11': vgg_model.create_vgg11_model,
    'vgg13': vgg_model.create_vgg13_model,
    'vgg16': vgg_model.create_vgg16_model,
    'vgg19': vgg_model.create_vgg19_model,
    'wide_resnet': wide_resnet_model.WideResnetModel,
    'efficientnet-b0': create_efficientnet_model,
    'efficientnet-b1': create_efficientnet_model,
    'efficientnet-b2': create_efficientnet_model,
    'efficientnet-b3': create_efficientnet_model,
    'efficientnet-b4': create_efficientnet_model,
    'efficientnet-b5': create_efficientnet_model,
    'efficientnet-b6': create_efficientnet_model,
    'resnet': resnet_model_imagenet.resnet,
}

_model_name_to_cifar_model = {
    'resnet': resnet_model_cifar.ResNet,
    'wide_resnet': wide_resnet_model.WideResnetModel,
    'diagonal_circulant': circulant_model.DiagonalCirculantModel,
    'scattering_circulant': scattering_model.ScatteringCirculantModel,
    'scattering_pooling_circulant':
      scattering_model.ScatteringPoolingCirculantModel,
    'scattering_by_channel_circulant':
      scattering_model.ScatteringByChannelCirculantModel,
    'ldr_model': structured_model.LDRModel,
    'ldr_multi_layer_model': structured_model.LDRMultiLayerModel
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



