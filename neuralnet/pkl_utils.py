"""
  utils functions for pickle package
"""
from os.path import isfile, splitext
import _pickle as pickle

def pickle_load(path):
  """ Function to load pickle object """
  with open(path, 'rb') as f:
    return pickle.load(f, encoding='latin1')

def _pickle_dump(file, path):
  """ Function to dump pickle object """
  with open(path, 'wb') as f:
    pickle.dump(file, f, -1)

def _get_new_name(path):
  """ Rename file if file already exist
    avoid erasing an existing file
  """
  i = 0
  new_path = path
  while isfile(new_path):
    ext = splitext(path)[1]
    new_path = path.replace(ext, '_{}{}'.format(i, ext))
    i += 1
  return new_path

def pickle_dump(file, path, force=False):
  """ Helper function to dump a file 
    without deleting an existing one
  """
  if force:
    _pickle_dump(file, path)
  elif not force:
    new_path = _get_new_name(path)
    _pickle_dump(file, new_path)
