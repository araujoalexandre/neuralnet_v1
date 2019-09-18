
import logging
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


class BaseReader:

  def __init__(self, params, batch_size, num_gpus, is_training):
    self.params = params
    self.num_gpus = num_gpus
    self.num_splits = num_gpus
    self.batch_size = batch_size
    self.batch_size_per_split = batch_size // self.num_splits
    self.is_training = is_training
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
      # TODO: fill up num_replicas and rank
      sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=None, rank=None)
    else:
      sampler = None
    loader = DataLoader(self.dataset,
                        batch_size=self.batch_size_per_split,
                        num_workers=self.num_threads,
                        shuffle=self.is_training and not sampler,
                        pin_memory=bool(self.num_gpus),
                        sampler=sampler)
    return loader



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

    self.dataset = MNIST(path, train=self.is_training,
                         download=False, transform=self.transform())

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
    self.n_classes = 10
    self.batch_shape = (None, 32, 32, 3)
    self.use_data_augmentation = self.params.data_augmentation

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
    self.dataset = CIFAR10(self.path, train=self.is_training,
                           download=False, transform=self.transform())


class CIFAR100Reader(CIFARReader):

  def __init__(self, params, batch_size, num_gpus, is_training):
    super(CIFAR100Reader, self).__init__(
      params, batch_size, num_gpus, is_training)
    self.dataset = CIFAR100(self.path, train=self.is_training,
                           download=False, transform=self.transform())


class IMAGENETReader(BaseReader):

  def __init__(self):
    super(IMAGENETReader, self).__init__(
      params, batch_size, num_gpus, is_training)

    # Provide square images of this size. 
    self.image_size = self.params.imagenet_image_size

    self.height, self.width = self.image_size, self.image_size
    self.n_train_files = 1281167
    self.n_test_files = 50000
    self.n_classes = 1001
    self.batch_shape = (None, self.height, self.height, 1)

    slpit = 'train' if self_is_training else 'val'
    self.dataset = ImageNet(self.path, slip=split,
                            download=False, transform=self.transform())

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
