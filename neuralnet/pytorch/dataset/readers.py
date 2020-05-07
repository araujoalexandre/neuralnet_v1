
import logging
import random
import math
from os.path import join
from os.path import exists

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from torchvision.datasets import ImageNet
from PIL import Image

from ..utils import AddNoise


class BaseReader:

  def __init__(self, params, batch_size, num_gpus, is_training):
    self.params = params
    self.num_gpus = num_gpus
    self.num_splits = num_gpus
    self.batch_size = batch_size
    self.is_training = is_training
    self.add_noise = getattr(
      self.params, 'add_noise', False) and self.is_training
    self.path = join(self.get_data_dir(), self.params.dataset)
    self.num_threads = self.params.datasets_num_private_threads

  def get_data_dir(self):
    paths = self.params.data_dir.split(':')
    data_dir = None
    for path in paths:
      if exists(join(path, self.params.dataset)):
        data_dir = path
        break
    if data_dir is None:
      raise ValueError("Data directory not found.")
    return data_dir

  def transform(self):
    """Create the transformer pipeline."""
    raise NotImplementedError('Must be implemented in derived classes')

  def load_dataset(self):
    """Load or download dataset."""
    if getattr(self.params, 'job_name', None):
      sampler = torch.utils.data.distributed.DistributedSampler(
        self.dataset, num_replicas=None, rank=None)
    else:
      sampler = None
    loader = DataLoader(self.dataset,
                        batch_size=self.batch_size,
                        num_workers=self.num_threads,
                        shuffle=self.is_training and not sampler,
                        pin_memory=bool(self.num_gpus),
                        sampler=sampler)
    return loader, sampler



class MNISTReader(BaseReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(MNISTReader, self).__init__(
      params, batch_size, num_gpus, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 28, 28
    self.n_train_files = 60000
    self.n_test_files = 10000
    self.n_classes = 10
    self.batch_shape = (None, 32, 32, 1)

    transform = self.transform()
    if self.add_noise:
      transform.transforms.append(AddNoise(self.params))

    self.dataset = MNIST(path, train=self.is_training,
                         download=False, transform=transform)

  def transform(self):
    transform = Compose([
        transforms.ToTensor()])
    return transform


class CIFARReader(BaseReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(CIFARReader, self).__init__(
      params, batch_size, num_gpus, is_training)

    self.batch_size = batch_size
    self.is_training = is_training
    self.height, self.width = 32, 32
    self.n_train_files = 50000
    self.n_test_files = 10000
    self.batch_shape = (None, 32, 32, 3)
    self.use_data_augmentation = self.params.data_augmentation

    self.cifar_mean = (0.4914, 0.4822, 0.4465) 
    self.cifar_std = (0.2023, 0.1994, 0.2010)

  def transform(self):
    if self.is_training:
      transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    else:
      transform = transforms.Compose([
        transforms.ToTensor()])
    return transform


class CIFAR10Reader(CIFARReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(CIFAR10Reader, self).__init__(
      params, batch_size, num_gpus, is_training)
    self.n_classes = 10

    transform = self.transform()
    if self.add_noise:
      transform.transforms.append(AddNoise(self.params))

    self.dataset = CIFAR10(self.path, train=self.is_training,
                           download=False, transform=transform)


class CIFAR100Reader(CIFARReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(CIFAR100Reader, self).__init__(
      params, batch_size, num_gpus, is_training)
    self.n_classes = 100

    transform = self.transform()
    if self.add_noise:
      transform.transforms.append(AddNoise(self.params))

    self.dataset = CIFAR100(self.path, train=self.is_training,
                           download=False, transform=transform)



class Lighting(object):
  """Lighting noise(AlexNet - style PCA - based noise)"""
  def __init__(self, alphastd, eigval, eigvec):
    self.alphastd = alphastd
    self.eigval = torch.Tensor(eigval)
    self.eigvec = torch.Tensor(eigvec)

  def __call__(self, img):
    if self.alphastd == 0:
      return img
    alpha = img.new().resize_(3).normal_(0, self.alphastd)
    rgb = self.eigvec.type_as(img).clone() \
      .mul(alpha.view(1, 3).expand(3, 3)) \
      .mul(self.eigval.view(1, 3).expand(3, 3)) \
      .sum(1).squeeze()
    return img.add(rgb.view(3, 1, 1).expand_as(img))


class EfficientNetRandomCrop:

    def __init__(self, imgsize, min_covered=0.1, 
                 aspect_ratio_range=(3./4, 4./3), area_range=(0.08, 1.0),
                 max_attempts=10):

      assert 0.0 < min_covered
      assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]
      assert 0 < area_range[0] <= area_range[1]
      assert 1 <= max_attempts

      self.min_covered = min_covered
      self.aspect_ratio_range = aspect_ratio_range
      self.area_range = area_range
      self.max_attempts = max_attempts
      self._fallback = EfficientNetCenterCrop(imgsize)

    def __call__(self, img):
      # https://github.com/tensorflow/tensorflow/blob/9274bcebb31322370139467039034f8ff852b004/tensorflow/core/kernels/sample_distorted_bounding_box_op.cc#L111
      original_width, original_height = img.size
      min_area = self.area_range[0] * (original_width * original_height)
      max_area = self.area_range[1] * (original_width * original_height)

      for _ in range(self.max_attempts):
        aspect_ratio = random.uniform(*self.aspect_ratio_range)
        height = int(round(math.sqrt(min_area / aspect_ratio)))
        max_height = int(round(math.sqrt(max_area / aspect_ratio)))

        if max_height * aspect_ratio > original_width:
          max_height = (original_width + 0.5 - 1e-7) / aspect_ratio
          max_height = int(max_height)
          if max_height * aspect_ratio > original_width:
            max_height -= 1

        if max_height > original_height:
          max_height = original_height

        if height >= max_height:
          height = max_height

        height = int(round(random.uniform(height, max_height)))
        width = int(round(height * aspect_ratio))
        area = width * height

        if area < min_area or area > max_area:
          continue
        if width > original_width or height > original_height:
          continue
        if area < self.min_covered * (original_width * original_height):
          continue
        if width == original_width and height == original_height:
          # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py#L102
          return self._fallback(img) 

        x = random.randint(0, original_width - width)
        y = random.randint(0, original_height - height)
        return img.crop((x, y, x + width, y + height))

      return self._fallback(img)


class EfficientNetCenterCrop:

    def __init__(self, imgsize):
      self.imgsize = imgsize

    def __call__(self, img):
      """Crop the given PIL Image and resize it to desired size.
      Args:
          img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
          output_size (sequence or int): (height, width) of the crop box. If int,
              it is used for both directions
      Returns:
          PIL Image: Cropped image.
      """
      image_width, image_height = img.size
      image_short = min(image_width, image_height)

      crop_size = float(self.imgsize) / (self.imgsize + 32) * image_short

      crop_height, crop_width = crop_size, crop_size
      crop_top = int(round((image_height - crop_height) / 2.))
      crop_left = int(round((image_width - crop_width) / 2.))
      return img.crop((crop_left, crop_top, crop_left + crop_width, crop_top + crop_height))






class IMAGENETReader(BaseReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(IMAGENETReader, self).__init__(
      params, batch_size, num_gpus, is_training)

    # Provide square images of this size. 
    self.image_size = self.params.imagenet_image_size
    if 'efficientnet' in self.params.model:
      self.image_size = {
        'efficientnet-b0': 224,
        'efficientnet-b1': 240,
        'efficientnet-b2': 260,
        'efficientnet-b3': 300,
        'efficientnet-b4': 380,
        'efficientnet-b5': 456,
        'efficientnet-b6': 528,
        'efficientnet-b7': 600,
      }[self.params.model]

    self.imagenet_pca = {
      'eigval': [0.2175, 0.0188, 0.0045],
      'eigvec': [
          [-0.5675,  0.7192,  0.4009],
          [-0.5808, -0.0045, -0.8140],
          [-0.5836, -0.6948,  0.4203],
      ]
    }

    self.height, self.width = self.image_size, self.image_size
    self.n_train_files = 1281167
    self.n_test_files = 50000
    self.n_classes = 1001
    self.batch_shape = (None, self.height, self.height, 1)

    split = 'train' if self.is_training else 'val'

    if 'efficientnet' in self.params.model:
      transform = self.efficientnet_transform()
    else:
      transform = self.transform()

    if self.add_noise:
      transform.transforms.append(AddNoise(self.params))

    self.dataset = ImageNet(self.path, split=split,
                            download=False, transform=transform)

  def efficientnet_transform(self):
    if self.is_training:
      transform = Compose([
          EfficientNetRandomCrop(self.image_size),
          transforms.Resize(
            (self.image_size, self.image_size), interpolation=Image.BICUBIC),
          transforms.RandomHorizontalFlip(),
          transforms.ColorJitter(
              brightness=0.4,
              contrast=0.4,
              saturation=0.4,
          ),
          transforms.ToTensor(),
          Lighting(0.1, self.imagenet_pca['eigval'], self.imagenet_pca['eigvec']),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
    else:
      transform = Compose([
          EfficientNetCenterCrop(input_size),
          transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      ])
    return transform

  def transform(self):
    if self.is_training:
      transform = Compose([
        transforms.Resize(self.image_size),
        transforms.CenterCrop(self.image_size),
        transforms.RandomResizedCrop(self.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])
    else:
      transform = Compose([
       transforms.Resize(self.image_size),
       transforms.CenterCrop(self.image_size),
       transforms.ToTensor()])
    return transform


readers_config = {
  'mnist': MNISTReader,
  'cifar10': CIFAR10Reader,
  'cifar100': CIFAR100Reader,
  'imagenet': IMAGENETReader
}
