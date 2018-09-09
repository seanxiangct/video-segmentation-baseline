
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import numpy as np

from TCN import TCN_LSTM
from utils import read_from_file, read_features, mask_data

local_feats_path = '/Users/seanxiang/data/cholec80/feats/'
remote_feats_path = '/home/cxia8134/dev/baseline/feats/'

model_name = 'TCN-LSTM-96,192nodes-'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=7, min_lr=0.5e-6, mode='auto')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
tensor_board = TensorBoard('log/' + model_name)

seq_length = 20
# n_nodes = [576, 288]
n_nodes = [64, 64]
nb_epoch = 100
nb_classes = 7
batch_size = 32
conv_len = [8, 16, 30, 64, 128][2]
n_feat = 2048
max_len = 6000

X_train, Y_train = read_features(local_feats_path, 'train')
X_vali, Y_vali = read_features(local_feats_path, 'vali')

X_train_m, Y_train_, M_train = mask_data(X_train, Y_train, max_len, mask_value=-1)
X_vali_m, Y_vali_, M_vali = mask_data(X_vali, Y_vali, max_len, mask_value=-1)

# ED-CNN
model = TCN_LSTM(n_nodes=n_nodes,
                 conv_len=conv_len,
                 n_classes=nb_classes,
                 n_feat=n_feat,
                 max_len=max_len,
                 online=False,
                 # activation='norm_relu',
                 return_param_str=False)

# train with extracted features from each video
model.fit(x=X_train_m,
          y=Y_train_,
          validation_data=(X_vali_m, Y_vali_, M_vali[:, :, 0]),
          epochs=nb_epoch,
          batch_size=16,
          verbose=1,
          sample_weight=M_train[:, :, 0],
          shuffle=False,
          callbacks=[lr_reducer, early_stopper, tensor_board])

model.save('trained/' + model_name + '.h5')
