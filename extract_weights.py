
from os.path import join
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

# path = '/pwrwork/rech/jvd/ubx31mc/models/2019-03-20_16.59.12.9736'
# path = '/pwrwork/rech/jvd/ubx31mc/models/2019-03-20_16.50.11.8975'
path = '/pwrwork/rech/jvd/ubx31mc/models/2019-03-20_17.54.37.0477'

np.set_printoptions(threshold=10000000, edgeitems=1000, precision=3,
                    suppress=True)

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file(
  join(path, "model.ckpt-750"),
  # tensor_name="tower/dense/kernel",
  tensor_name="tower/network_defense/dense/kernel",
  all_tensors=False)
