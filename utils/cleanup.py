import os
import argparse
import glob
from os.path import join, exists

def clean_up(folder_to_clean):

  path = join(os.environ.get('WORKDIR'), folder_to_clean)
  logs_folder = glob.glob(join(path, '**_logs'))

  models = []
  for path_logs_folder in logs_folder:
    best_accuracy_file = join(path_logs_folder, 'best_accuracy.txt')
    path_folder = path_logs_folder.strip('_logs')
    if exists(best_accuracy_file):
      with open(best_accuracy_file) as f:
        best_model_id, acc = f.read().split('\t')
        models.append((path_folder, best_model_id)) 
    else:
      models.append((path_folder, None))

  for path, best_model_id in models:
    if best_model_id is not None:
      best_model_name = 'model.ckpt-{}.pth'.format(best_model_id)
      checkpoints = glob.glob(join(path, '*.pth'))
      for ckpt in checkpoints:
        if ckpt.split('/')[-1] != best_model_name:
          os.remove(ckpt)
    else:
      print('this folder {} has no best model id'.format(path))



if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Script to clean up checkpoints.')
  parser.add_argument("--folder", type=str,
                        help="Name of folder to clean up.")
  args = parser.parse_args()
  clean_up(args.folder)

