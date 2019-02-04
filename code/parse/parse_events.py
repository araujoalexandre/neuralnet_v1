
import os
import sys
import argparse
from os.path import join, dirname
from collections import defaultdict
from glob import glob
import tensorflow as tf

def parse_events(events_path, dataset='train'):
  folder = dirname(events_path)
  data = defaultdict(dict)
  best_acc, best_step = -1, None
  for event in tf.train.summary_iterator(events_path):
    if event.step:
      loss, acc = None, None
      for v in event.summary.value:
        for metric in ['epoch', 'loss', 'accuracy']:
          if v.tag == metric:
            value = v.simple_value
            data[event.step][metric] = value
            if metric == 'accuracy' and value > best_acc:
              best_acc, best_step = value, event.step

  output = join("{}_logs".format(folder),
                "table_eval_{}.csv".format(dataset))
  with open(output, 'w') as f:
    f.write("step\tepochs\tloss\taccuracy\n")
    for step in sorted(data.keys()):
      val = data[step]
      epoch, loss, acc = val["epoch"], val["loss"], val["accuracy"]
      f.write("{}\t{:.2f}\t{:.4f}\t{:.4f}\n".format(step, epoch, loss, acc))

  if dataset == "test":
    output = join("{}_logs".format(folder), "best_accuracy.txt")
    with open(output, 'w') as f:
      f.write("{}\t{:.4f}\n".format(best_step, best_acc))

  if dataset == "train":
    output = join("{}_logs".format(folder), "best_accuracy_train.txt")
    with open(output, 'w') as f:
      f.write("{}\t{:.4f}\n".format(best_step, best_acc))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
              description="Script to parse log file.")
  parser.add_argument("--folder", type=str, help="folder of experiment.")
  args = parser.parse_args()

  events_files = glob(join(args.folder, "events*"))
  events_path = {}
  for files in events_files:
    if "eval_train" in files:
      events_path["eval_train"] = files
    if "eval_test" in files:
      events_path["eval_test"] = files

  try:
    parse_events(events_path["eval_train"], dataset='train')
  except Exception as e:
    tf.logging.info("error parsing eval_train: {}".format(e))

  try:
    parse_events(events_path["eval_test"], dataset='test')
  except Exception as e:
    tf.logging.info("error parsing eval_test: {}".format(e))

