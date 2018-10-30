import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from keras import Input, Model, regularizers
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
import numpy as np
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense, Permute, Lambda, K, RepeatVector, multiply, \
    Flatten, CuDNNLSTM, Softmax, Multiply

from modules.utils import read_from_file, read_features, mask_data, phase_length, sample_weights, cal_avg_len, \
    train_generator, vali_generator
from sklearn.utils import class_weight

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

local_feats_path = '/Users/seanxiang/data/cholec80/feats/'
remote_feats_path = '/home/cxia8134/dev/baseline/feats/'

model_name = 'BiLSTM-500nodes-noMask-densenetFeats-8'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=7, min_lr=0.5e-6, mode='auto')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
tensor_board = TensorBoard('log/' + model_name)
# save model if validation loss decreased
checkpointer = ModelCheckpoint(filepath='/home/cxia8134/dev/baseline/temp/' + model_name
                                        + '-{epoch:02d}-{val_loss:.2f}.hdf5',
                               verbose=1,
                               save_best_only=True)


def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    # a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(n_nodes, activation='softmax', name='weighting')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs], name='attention_mul')
    return output_attention_mul


def attention_block(inputs, length):
    lstm = Bidirectional(LSTM(length, return_sequences=True,
                              dropout=0.5,
                              recurrent_dropout=0.5,
                              kernel_regularizer=regularizers.l2(l2_norm)))(inputs)
    attention = TimeDistributed(Dense(1))(lstm)
    attention = Softmax(axis=1, name='attention_vec')(attention)
    context = Multiply(name='attention_mul')([attention, lstm])
    return context


n_nodes = 500
nb_epoch = 200
nb_classes = 7
batch_size = 1
n_feat = 2048
max_len = 6000
l2_norm = 0.01
attention_length = 100
feats = ['resnet', 'densenet'][1]
SINGLE_ATTENTION_VECTOR = False

path = remote_feats_path

X_train, Y_train = read_features(path, feats, 'train')
X_vali, Y_vali = read_features(path, feats, 'vali')

# TODO: append frame id to feature

# X_train_m, Y_train_, M_train = mask_data(X_train, Y_train, max_len, mask_value=-1)
# X_vali_m, Y_vali_, M_vali = mask_data(X_vali, Y_vali, max_len, mask_value=-1)

# calculate sample weights based on ground-truth label distributions
# sample_weights = sample_weights(Y_train)

# find the average length of the training samples
# avg_len = cal_avg_len(X_train)

inputs = Input(shape=(None, n_feat))

# model = attention_block(inputs, 100)

model = Bidirectional(LSTM(n_nodes,
                           return_sequences=True,
                           input_shape=(batch_size, None, n_feat),
                           dropout=0.5,
                           name='bilstm',
                           recurrent_dropout=0.25))(inputs)
# model = LSTM(n_nodes,
#              return_sequences=True,
#              input_shape=(batch_size, None, n_feat),
#              dropout=0.5,
#              name='bilstm',
#              recurrent_dropout=0.25)(inputs)

# attention layer
# model = attention_3d_block(model)

# Output FC layer
model = TimeDistributed(Dense(nb_classes, activation="softmax"))(model)

model = Model(inputs=inputs, outputs=model)
# model = multi_gpu_model(model, gpus=2)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              sample_weight_mode="temporal",
              metrics=['accuracy'])
model.summary()

# train on videos with sample weighting
# model.fit(x=X_train_m,
#           y=Y_train_,
#           validation_data=(X_vali_m, Y_vali_, M_vali[:, :, 0]),
#           epochs=nb_epoch,
#           batch_size=batch_size,
#           verbose=1,
#           # sample_weight=M_train[:, :, 0],
#           sample_weight=sample_weights,
#           callbacks=[lr_reducer, early_stopper, tensor_board, checkpointer])


model.fit_generator(train_generator(X_train, Y_train),
                    verbose=1,
                    epochs=nb_epoch,
                    steps_per_epoch=50,
                    validation_steps=10,
                    validation_data=vali_generator(X_vali, Y_vali),
                    callbacks=[lr_reducer, early_stopper, tensor_board, checkpointer])

model.save('trained/' + model_name + '.h5')
