# TensorFlow Research Library to train Deep Neural Network with structured matrices


'''
mkdir data
python3 code/generate_tfrecords.py --output_dir=`pwd`/data
'''

'''
python3 code/train.py --data_dir=`pwd`/data --start_new_model --log_steps=500 --reader=CIFAR10Reader --model=Cifar10ModelToeplitz
'''

