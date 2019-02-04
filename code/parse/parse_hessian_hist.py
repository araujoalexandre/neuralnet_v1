import os
import sys
import re
import argparse
from os.path import join, dirname
from collections import defaultdict
from glob import glob
import tensorflow as tf

def parse_events(events_path):
  folder = dirname(events_path)
  data = {}
  gradients_norm = defaultdict(dict)
  try:
    for event in tf.train.summary_iterator(events_path):
      count = 0
      if event.step:
        for v in event.summary.value:
          if "gradients/norm" in v.tag:
            if v.simple_value <= 0.03:
              count += 1
        print(event.step, count)

  except tf.errors.DataLossError:
    pass

  k0 = sorted(gradients_norm.keys())[0]
  keys = sorted(gradients_norm[k0].keys())
  count = 0
  for step in sorted(gradients_norm.keys()):
    step = int(step)
    for key in keys:
      value = gradients_norm[step][key]
      if value <= 0.03:
        count += 1
  print('count', count)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
              description="Script to parse log file.")
  parser.add_argument("--folder", type=str, help="folder of experiment.")
  args = parser.parse_args()

  event_files = glob(join(args.folder, "events*"))
  for file_ in event_files:
    if "eval" not in file_:
      path = file_
  parse_events(path)

