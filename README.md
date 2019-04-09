# TensorFlow Research Library

Setup library

```
git clone https://github.com/araujoalexandre/neuralnet.git
export PROJECTDIR=`pwd`/neuralnet

# define the data folder
mkdir /path/to/data
export DATADIR=/path/to/data
# download and processed datasets 
python3 code/generate_tfrecords.py --output_dir=$DATADIR

# define the models folder
export WORKDIR=/path/to/models
```
