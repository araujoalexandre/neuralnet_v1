
import time
import pprint
import socket
import logging
import warnings
from os import mkdir
from os.path import join
from os.path import exists

import utils as global_utils
from . import utils
from .models import model_config

from .dataset.readers import readers_config

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler


def get_scheduler(optimizer, policy="multistep",
                  milestones=[60, 120, 160], gamma=0.2):
  """Return a learning rate scheduler
  Parameters:
      optimizer          -- the optimizer of the network
      opt (option class) -- stores all the experiment flags; needs to be a
                            subclass of BaseOptions. opt.lr_policy is the name
                            of learning rate policy:
                            linear | step | plateau | cosine
  For 'linear', we keep the same learning rate for the first <opt.niter> epochs
  and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
  For other schedulers (step, plateau, and cosine), we use the default PyTorch
  schedulers. See https://pytorch.org/docs/stable/optim.html for more details.
  """
  if policy == 'linear':
      def lambda_rule(epoch):
          lr_l = 1.0 - max(0, epoch + opt.start_epoch -
                           opt.niter) / float(opt.niter_decay + 1)
          return lr_l
      scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
  elif policy == 'step':
      scheduler = lr_scheduler.StepLR(
          optimizer, step_size=milestones, gamma=gamma)
  elif policy == 'multistep':
      scheduler = lr_scheduler.MultiStepLR(
          optimizer,   milestones=milestones, gamma=gamma, last_epoch=-1)
  else:
      return NotImplementedError(
        "learning rate policy [%s] is not implemented", policy)
  return scheduler


class Trainer:
  """A Trainer to train a PyTorch."""

  def __init__(self, params):
    """Creates a Trainer.
    """
    utils.set_default_param_values_and_env_vars(params)
    self.params = params

    # Setup logging & log the version.
    global_utils.setup_logging(params.logging_verbosity)
    logging.info("PyTorch version: {}.".format(torch.__version__))
    logging.info("Hostname: {}.".format(socket.gethostname()))

    # print self.params parameters
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(params.values()))

    self.job_name = self.params.job_name  # "" for local training
    self.task_index = self.params.task_index
    self.is_master = (self.job_name in ('', 'worker') and self.task_index == 0)
    self.start_new_model = self.params.start_new_model
    self.train_dir = self.params.train_dir
    self.num_gpus = self.params.num_gpus
    self.batch_size = self.params.batch_size * self.num_gpus

    # create a mesage builder for logging
    self.message = global_utils.MessageBuilder()

    # load reader and model
    self.reader = readers_config[self.params.dataset](
      self.params, self.batch_size, self.num_gpus, is_training=True)
    self.model = model_config.get_model_config(
        self.params.model, self.params.dataset, self.params,
        self.reader.n_classes, is_training=True)
    if self.num_gpus:
      self.model = torch.nn.DataParallel(self.model).cuda()


  def run(self):
    """Performs training on the currently defined Tensorflow graph.
    """
    # reset the training directory if start_new_model is True
    if self.is_master and self.start_new_model and exists(self.train_dir):
      global_utils.remove_training_directory(self.train_dir)
    mkdir(self.train_dir)



    if self.params.torch_random_seed is not None:
      random.seed(self.params.torch_random_seed)
      torch.manual_seed(self.params.torch_random_seed)
      cudnn.deterministic = True
      warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    if self.params.cudnn_benchmark:
      cudnn.benchmark = True


    # # save the parameters in json formet in the training directory
    # model_flags_dict = self.params.to_json()
    # log_folder = '{}_logs'.format(self.train_dir)
    # flags_json_path = join(log_folder, "model_flags.json")
    # if not exists(flags_json_path):
    #   with open(flags_json_path, "w") as fout:
    #     fout.write(model_flags_dict)

    self._run_training()


  def _run_training(self):

    self.criterion = torch.nn.CrossEntropyLoss().cuda()
    self.optimizer = torch.optim.SGD(
                       self.model.parameters(),
                       lr=0.01,
                       momentum=0.9,
                       weight_decay=5e-4)
    scheduler = get_scheduler(
      self.optimizer, policy="multistep",
      milestones=[30, 60, 90], gamma=0.2)

    batch_size = self.batch_size
    n_files = self.reader.n_train_files

    logging.info("Start training")
    global_step = 0
    for _ in range(self.params.num_epochs):
      for data in self.reader.load_dataset():
        epoch = ((global_step * batch_size) / n_files)
        self._training(data, epoch, global_step)
        self.save_ckpt(global_step, epoch)
        global_step += 1
      scheduler.step()
    logging.info("Done training -- epoch limit reached.")

  def save_ckpt(self, step, epoch):
    """Save ckpt in train directory."""
    if step % self.params.save_checkpoint_steps == 0 and self.is_master:
        state = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
        }
        ckpt_name = "model.ckpt-{}.t7".format(step)
        logging.info("Saving checkpoint '{}'.".format(ckpt_name))
        torch.save(state, join(self.train_dir, ckpt_name))


  def _training(self, data, epoch, step):

    batch_start_time = time.time()
    inputs, labels = data
    outputs = self.model(inputs)
    self.optimizer.zero_grad()
    loss = self.criterion(outputs, labels.cuda())
    loss.backward()
    self.optimizer.step()
    seconds_per_batch = time.time() - batch_start_time
    examples_per_second = self.batch_size / seconds_per_batch

    to_print = step % self.params.frequency_log_steps == 0
    if (self.is_master and to_print) or step == 1:
      lr = self.optimizer.param_groups[0]['lr']
      self.message.add("epoch", epoch, format="4.2f")
      self.message.add("step", step, width=5, format=".0f")
      self.message.add("lr", lr, format=".6f")
      self.message.add("loss", loss, format=".4f")
      self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
      logging.info(self.message.get_message())


