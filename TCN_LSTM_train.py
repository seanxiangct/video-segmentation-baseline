from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import numpy as np

from TCN import TCN_LSTM, residual_TCN_LSTM, ED_TCN, attention_TCN_LSTM
from modules.utils import read_from_file, read_features, mask_data, phase_length
from sklearn.utils import class_weight

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))

local_feats_path = '/Users/seanxiang/data/cholec80/feats/'
remote_feats_path = '/home/cxia8134/dev/baseline/feats/'

model_name = 'attention-TCNLSTM-64,128,256nodes-16conv-sampleweights-4'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=7, min_lr=0.5e-6, mode='auto')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
tensor_board = TensorBoard('log/' + model_name)

n_nodes = [64, 128, 256]
nb_epoch = 100
nb_classes = 7
batch_size = 4
conv_len = [8, 16, 32, 64, 128][1]
n_feat = 2048
max_len = 6000

path = remote_feats_path

X_train, Y_train = read_features(path, 'train')
X_vali, Y_vali = read_features(path, 'vali')

# TODO: append frame id to feature

X_train_m, Y_train_, M_train = mask_data(X_train, Y_train, max_len, mask_value=-1)
X_vali_m, Y_vali_, M_vali = mask_data(X_vali, Y_vali, max_len, mask_value=-1)

# split training data and its labels in half and apply sample weight to half of it
# print(X_train_m.shape)
# print(Y_train_.shape)
# print(M_train.shape)
# print(M_train[:(len(M_train) // 2), :, 0].shape)

# data train without sample weight
# X_train_1 = X_train_m[:(len(X_train_m) // 2), :, :]
# Y_train_1 = Y_train_[:(len(Y_train_) // 2), :, :]

# data train with sample weight
# X_train_2 = X_train_m[(len(X_train_m) // 2):, :, :]
# Y_train_2 = Y_train_[(len(Y_train_) // 2):, :, :]


# TODO: address the imbalanced class issue
# iterate through the unmasked ground truth labels
sample_weights = []
for sample in Y_train:

    labels = np.array([np.argmax(y, axis=None, out=None) for y in sample])
    # weights for each class
    weights = class_weight.compute_class_weight('balanced',
                                                np.unique(labels),
                                                labels)

    # an array has the length with the video
    frame_weights = []
    # find the length of each phase segments is an array with the length of the number of classes and containing the
    # values corresponding (length, label) to each class
    segments = phase_length(labels)

    # add frame weighting based on frame phases
    for i, s in enumerate(segments):
        length = s[0]
        label = s[1]
        weight = 0
        if len(segments) == len(weights):
            weight = weights[i]
        elif len(segments) > len(weights):
            # when phases going back and forth
            try:
                weight = weights[label]
            except IndexError:
                weight = weights[i]

        phase = [weight] * length
        frame_weights.extend(phase)

    # add mask weighting
    if len(frame_weights) != max_len:
        frame_weights.extend(np.zeros(max_len - len(frame_weights)))

    sample_weights.append(np.array(frame_weights))

sample_weights = np.array(sample_weights)

# ED-CNN
model = attention_TCN_LSTM(n_nodes=n_nodes,
                           conv_len=conv_len,
                           n_classes=nb_classes,
                           n_feat=n_feat,
                           max_len=max_len,
                           online=False,
                           return_param_str=False)

# train with extracted features from each video
# start with video without sample weighting
# model.fit(x=X_train_1,
#           y=Y_train_1,
#           validation_data=(X_vali_m, Y_vali_, M_vali[:, :, 0]),
#           epochs=nb_epoch,
#           batch_size=batch_size,
#           verbose=1,
#           sample_weight=M_train[:(len(M_train) // 2), :, 0],
#           callbacks=[lr_reducer, early_stopper, tensor_board])

# train on videos with sample weighting
model.fit(x=X_train_m,
          y=Y_train_,
          validation_data=(X_vali_m, Y_vali_, M_vali[:, :, 0]),
          epochs=nb_epoch,
          batch_size=batch_size,
          verbose=1,
          # sample_weight=M_train[:, :, 0],
          sample_weight=sample_weights,
          callbacks=[lr_reducer, early_stopper, tensor_board])

model.save('trained/' + model_name + '.h5')
