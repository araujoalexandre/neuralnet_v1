
import torch
import torch.nn as nn
from torch.autograd import Variable


def conv_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    init.xavier_uniform_(m.weight, gain=np.sqrt(2))
    init.constant_(m.bias, 0)


def cfg(depth):
  depth_lst = [11, 13, 16, 19]
  assert (depth in depth_lst), "Error : VGGnet depth should be either 11, 13, 16, 19"
  cf_dict = {
    11: [
      64, 'mp',
      128, 'mp',
      256, 256, 'mp',
      512, 512, 'mp',
      512, 512, 'mp'],
    13: [
      64, 64, 'mp',
      128, 128, 'mp',
      256, 256, 'mp',
      512, 512, 'mp',
      512, 512, 'mp'],
    16: [
      64, 64, 'mp',
      128, 128, 'mp',
      256, 256, 256, 'mp',
      512, 512, 512, 'mp',
      512, 512, 512, 'mp'],
    19: [
      64, 64, 'mp',
      128, 128, 'mp',
      256, 256, 256, 256, 'mp',
      512, 512, 512, 512, 'mp',
      512, 512, 512, 512, 'mp'],
  }
  return cf_dict[depth]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class VGG(nn.Module):

  def __init__(self, params, depth, num_classes):
    super(VGG, self).__init__()
    self.features = self._make_layers(cfg(depth))
    self.linear = nn.Linear(512, num_classes)

  def _make_layers(self, cfg):
    layers = []
    in_planes = 3
    for x in cfg:
      if x == 'mp':
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
      else:
        layers += [conv3x3(in_planes, x), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
        in_planes = x
    # After cfg convolution
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.features(x)
    out = out.view(out.size(0), -1)
    out = self.linear(out)
    return out


def create_vgg11_model(params, nclass):
  return VGG(params, 11, nclass)

def create_vgg13_model(params, nclass):
  return VGG(params, 13, nclass)

def create_vgg16_model(params, nclass):
  return VGG(params, 16, nclass)

def create_vgg19_model(params, nclass):
  return VGG(params, 19, nclass)




