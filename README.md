# TensorFlow Research Library

Setup library

```
git clone https://github.com/araujoalexandre/neuralnet.git
export PROJECTDIR=`pwd`/neuralnet

# define the data and models folders
mkdir data models
export DATADIR=`pwd`/data
export WORKDIR=`pwd`

# download and processed datasets 
cd neuralnet
python3 code/dataset/generate_tfrecords.py --output_dir=$DATADIR

python3 sub/train.py config_name | bash
```
