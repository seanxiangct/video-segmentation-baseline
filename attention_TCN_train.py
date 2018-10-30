import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np

from TCN import TCN_LSTM, residual_TCN_LSTM, ED_TCN, attention_TCN_LSTM, ED_TCN
from modules.utils import read_from_file, read_features, mask_data, phase_length, train_generator, vali_generator
from sklearn.utils import class_weight

from keras.utils import multi_gpu_model

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

local_feats_path = '/Users/seanxiang/data/cholec80/feats/'
remote_feats_path = '/home/cxia8134/dev/baseline/feats/'

model_name = 'TCN-576,288nodes-32conv-noMask-attention-online-2'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=7, min_lr=0.5e-6, mode='auto')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
tensor_board = TensorBoard('log/' + model_name)
# save model if validation loss decreased
checkpointer = ModelCheckpoint(filepath='/home/cxia8134/dev/baseline/temp/{epoch:02d}-{val_loss:.2f}.hdf5',
                               verbose=1,
                               save_best_only=True)

n_nodes = [576, 288]
nb_epoch = 200
nb_classes = 7
batch_size = 10
conv_len = [8, 16, 32, 64, 128][2]
n_feat = 2048
max_len = 6000

path = remote_feats_path

X_train, Y_train = read_features(path, 'train')
X_vali, Y_vali = read_features(path, 'vali')

# TODO: append frame id to feature

# X_train_m, Y_train_, M_train = mask_data(X_train, Y_train, max_len, mask_value=-1)
# X_vali_m, Y_vali_, M_vali = mask_data(X_vali, Y_vali, max_len, mask_value=-1)


# TODO: address the imbalanced class issue
# iterate through the unmasked ground truth labels

# ED-CNN
model = ED_TCN(n_nodes=n_nodes,
               conv_len=conv_len,
               n_classes=nb_classes,
               n_feat=n_feat,
               max_len=max_len,
               optimizer='adam',
               attention=True,
               online=True,
               return_param_str=False)

# train on videos with sample weighting
model.fit_generator(train_generator(X_train, Y_train),
                    verbose=1,
                    epochs=nb_epoch,
                    steps_per_epoch=50,
                    validation_steps=10,
                    validation_data=vali_generator(X_vali, Y_vali),
                    callbacks=[lr_reducer, early_stopper, tensor_board, checkpointer])

model.save('trained/' + model_name + '.h5')
