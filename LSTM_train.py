from keras import Input, Model, Sequential
from keras.layers import LSTM, Bidirectional, TimeDistributed, Dense
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import numpy as np

import seq2seq
from seq2seq.models import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq

from TCN import ED_TCN
from utils.utils import read_from_file, read_features, mask_data

remote_feats_path = '/home/cxia8134/dev/baseline/feats/'
remote_train_path = '/home/cxia8134/dev/baseline/feats/train'
remote_vali_path = '/home/cxia8134/dev/baseline/feats/vali'

model_name = 'BidirectionalLSTM-encoder-decoder-without_peek-1'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6, mode='auto')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
tensor_board = TensorBoard('log/' + model_name)

seq_length = 20
n_nodes = [96, 192]
nb_epoch = 100
nb_classes = 7
batch_size = 32
conv_len = [8, 16, 32, 64, 128][2]

n_timesteps_in = 30
n_feat = 2048
max_len = 6000

X_train, Y_train = read_features(remote_feats_path, 'train')
X_vali, Y_vali = read_features(remote_feats_path, 'vali')

X_train_m, Y_train_, M_train = mask_data(X_train, Y_train, max_len, mask_value=-1)
X_vali_m, Y_vali_, M_vali = mask_data(X_vali, Y_vali, max_len, mask_value=-1)

# use validation split provided by keras.model.fit instead
# X_train_m = np.vstack((X_train_m, X_vali_m))
# Y_train_ = np.vstack((Y_train_, Y_vali_))

# bidirectional LSTM
# model = Sequential()
# model.add(Bidirectional(LSTM(500, return_sequences=True,
#                              dropout=0.25,
#                              recurrent_dropout=0.25,
#                              input_shape=(n_timesteps_in, n_feat))))
# model.add(TimeDistributed(Dense(nb_classes, activation="softmax")))
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               sample_weight_mode='temporal',
#               metrics=['acc'])

model = AttentionSeq2Seq(input_dim=n_feat,
                         input_length=max_len,
                         hidden_dim=20,
                         output_length=max_len,
                         output_dim=nb_classes,
                         depth=4)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              sample_weight_mode='temporal',
              metrics=['acc'])
model.summary()

# train with extracted features from each video
model.fit(x=X_train_m,
          y=Y_train_,
          validation_data=(X_vali_m, Y_vali_, M_vali[:, :, 0]),
          # validation_split=0.1,
          epochs=nb_epoch,
          batch_size=5,
          verbose=1,
          sample_weight=M_train[:, :, 0],
          shuffle=False,
          callbacks=[lr_reducer, early_stopper, tensor_board])

model.save('trained/' + model_name + '.h5')
