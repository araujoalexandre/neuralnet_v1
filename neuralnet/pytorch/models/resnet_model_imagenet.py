
from torchvision import models

def resnet(params, num_classes, is_training):
  if params.model_params['depth'] == 18:
    return models.resnet18()
  elif params.model_params['depth'] == 34:
    return models.resnet34()
  elif params.model_params['depth'] == 50:
    return models.resnet50()
  elif params.model_params['depth'] == 101:
    return models.resnet101()
  elif params.model_params['depth'] == 152:
    return models.resnet152()
  else:
    raise ValueError('Depth not recopgbnized.')
