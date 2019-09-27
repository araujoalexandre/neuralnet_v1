
import json
import time
import os
import sys
import re
import socket
import pprint
import logging
import argparse
from os.path import join
from os.path import exists

import utils as global_utils
from . import utils
from .models import model_config

from .dataset.readers import readers_config

import numpy as np
import torch


class EvaluatorEnsemble:
  """Make an Ensemble of Pytorch Models."""

  def __init__(self, params):

    # Set up environment variables before doing any other global initialization to
    # make sure it uses the appropriate environment variables.
    utils.set_default_param_values_and_env_vars(params)

    self.params = params

    # Setup logging & log the version.
    global_utils.setup_logging(params.logging_verbosity)
    logging.info("Pytorch version: {}.".format(torch.__version__))
    logging.info("Hostname: {}.".format(socket.gethostname()))

    # print self.params parameters
    pp = pprint.PrettyPrinter(indent=2, compact=True)
    logging.info(pp.pformat(params.values()))

    self.num_gpus = self.params.num_gpus

    # create a mesage builder for logging
    self.message = global_utils.MessageBuilder()

    if self.params.num_gpus:
      self.batch_size = self.params.batch_size * self.num_gpus
    else:
      self.batch_size = self.params.batch_size

    if not self.params.data_pattern:
      raise IOError("'data_pattern' was not specified. "
        "Nothing to evaluate.")

    # load reader and model
    self.reader = readers_config[self.params.dataset](
      self.params, self.batch_size, self.num_gpus, is_training=False)
    self.model = model_config.get_model_config(
        self.params.model, self.params.dataset, self.params,
        self.reader.n_classes, is_training=False)
    # TODO: get the loss another way
    self.criterion = torch.nn.CrossEntropyLoss().cuda()

    if self.num_gpus:
      # self.model = torch.nn.DataParallel(self.model).cuda()
      self.model = self.model.cuda()


  def run(self):
    """Evaluate a n models for Ensemble
    """
    logging.info("Building evaluation graph")
    logits = []
    for i, folder in enumerate(self.params.folders):
      logs_dir = '{}_logs'.format(folder)
      best_checkpoint, global_step = \
          global_utils.get_best_checkpoint(
            logs_dir, backend='pytorch')
      checkpoint = torch.load(best_checkpoint)
      global_step = checkpoint['global_step']
      epoch = checkpoint['epoch']
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.model.eval()
      logits_model, labels = self.eval_loop(global_step, epoch)
      logits.append(logits_model)

    logits = np.mean(logits, axis=0)
    hard_pred = np.argmax(logits, axis=1)
    accuracy = (labels == hard_pred).mean()
    logging.info("accuracy on {} models: {:.6f}".format(
      len(self.params.folders), accuracy))
    logging.info("Done evaluation -- number of eval reached.")


  def eval_loop(self, global_step, epoch):
    """Run the evaluation loop once."""

    running_accuracy = 0
    running_inputs = 0
    running_loss = 0
    labels_list, logits_list = [], []
    for batch_n, data in enumerate(self.reader.load_dataset()):

      with torch.no_grad():
        batch_start_time = time.time()
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        seconds_per_batch = time.time() - batch_start_time
        examples_per_second = inputs.size(0) / seconds_per_batch
        logits_list.append(outputs.cpu().numpy())
        labels_list.append(labels.cpu().numpy())

      running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
      running_inputs += inputs.size(0)
      running_loss += loss.cpu().numpy()
      accuracy = running_accuracy / running_inputs
      loss = running_loss / (batch_n + 1)

      self.message.add('step', global_step)
      self.message.add('accuracy', accuracy, format='.5f')
      self.message.add('loss', loss, format='.5f')
      self.message.add('imgs/sec', examples_per_second, format='.0f')
      logging.info(self.message.get_message())

    self.message.add('step', global_step)
    self.message.add('accuracy', accuracy, format='.5f')
    self.message.add('loss', loss, format='.5f')
    logging.info(self.message.get_message())
    logging.info("Done with batched inference.")
    return np.concatenate(logits_list), np.concatenate(labels_list)








