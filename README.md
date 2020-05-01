# TensorFlow & PyTorch Research Library

## Setup library

```
git clone https://github.com/araujoalexandre/neuralnet.git
export PROJECTDIR=`pwd`/neuralnet

# define the data and models folders
mkdir data models
export DATADIR=`pwd`/data
export WORKDIR=`pwd`

# download and processed datasets 
cd neuralnet
python3 code/dataset/generate_tfrecords.py --output_dir=$DATADIR --dataset=all
```


## Run training and eval

The config.yaml file should be in the config folder
```
./sub/submit.py --config=config --cluster=None | bash
```

You may have to choose the backend given the model to use
```
./sub/submit.py --config=config_torch --backend=pytorch --cluster=None | bash
```


To make an eval under attack after training
```
./sub/submit.py --mode=attack --attack=fgm --folder=XXX --cluster=None | bash 
```
Attack choice: fgm, pgd, carlini and elasticnet



## Grid Search

To launch a series of experiment with different parameters, it possible to use the config file as a template and populate it with values. 

Example of config file as template:
```
default: &DEFAULT
  train_batch_size:           {batch_size}
  num_epochs:                 {epochs}
  start_new_model:            True
  train_num_gpu:              2
...
```

You can populate the values with the "params" parameter:
```
./sub/submit.py --config=config --cluster=None --grid_search="epochs:5,10,15;batch_size:32,64,128}'
```

New config file will be generated and saved in the 'config_gen' folder. Each config file will have a different id. 































