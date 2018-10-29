from keras import Model, Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.models import Sequential, load_model
import numpy as np
from keras.layers import Bidirectional, LSTM, TimeDistributed, Dense, Flatten
from utils.DataGenerator import DataGenerator
from utils.utils import read_from_file

remote_train_pair = '/home/cxia8134/data/train_labels/labels.txt'
remote_vali_pair = '/home/cxia8134/data/vali_labels/labels.txt'

remote_model_path = '/home/cxia8134/dev/baseline/trained/baseline_1.h5'
local_model_path = '/Users/seanxiang/data/trained/baseline_1.h5'

model_name = 'CNN-BiLSTM-1'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=8, min_lr=0.5e-6, mode='auto')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10)
tensor_board = TensorBoard('log/' + model_name)

input_height, input_width = 224, 224
input_channels = 3

seq_length = 20
n_nodes = 200
nb_epoch = 200
timesteps = 8
nb_classes = 7
batch_size = 32
n_train = 86344
n_vali = 21108

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.

pretrained = load_model(remote_model_path)
# remote the last prediction layer
pretrained.layers.pop()

# flatten the maxpooled feature tensors into feature vectors
# flattened = Flatten()(pretrained.layers[-1])
n_feats = pretrained.output_shape

# bidirectional lstm
bidirectional = Bidirectional(LSTM(n_nodes,
                                   return_sequences=True,
                                   input_shape=(None, seq_length, n_feats),
                                   dropout=0.5,
                                   recurrent_dropout=0.25))(pretrained)
bidirectional = TimeDistributed(Dense(nb_classes, activation='softmax'))(bidirectional)

model = Model(input=pretrained.layers[0], outputs=bidirectional)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              sample_weight_mode="temporal", metrics=['accuracy'])
model.summary()

train_pair = read_from_file(remote_train_pair)
vali_pair = read_from_file(remote_vali_pair)
train_generator = DataGenerator(train_pair, nb_classes, batch_size)
vali_generator = DataGenerator(vali_pair, nb_classes, batch_size)

model.fit_generator(generator=train_generator,
                    steps_per_epoch=(n_train // batch_size),
                    epochs=nb_epoch,
                    validation_data=vali_generator,
                    validation_steps=(n_vali // batch_size),
                    verbose=1,
                    use_multiprocessing=True,
                    workers=8,
                    max_queue_size=32,
                    callbacks=[lr_reducer, early_stopper, tensor_board])

model.save('trained/' + model_name + '.h5')

# return sequence:
# true: return output for every node
# false: only return output for the last node

# input shape of LSTM: (sequence length, time steps, input features)
# batch_size: size of the batches during training
# time steps: number of previous inputs being referenced (setting to None to accept variable length input)
# input features: inputs feature length
# model.add(LSTM(32, return_sequences=True, stateful=True,
#                batch_input_shape=(batch_size, timesteps, data_dim)))
# model.add(LSTM(32, return_sequences=True, stateful=True))
# model.add(LSTM(32, stateful=True))
# model.add(Dense(10, activation='softmax'))
#
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])
# model.summary()
#
# # Generate dummy training data
# x_train = np.random.random((batch_size * 10, timesteps, data_dim))
# y_train = np.random.random((batch_size * 10, num_classes))
#
# # Generate dummy validation data
# x_val = np.random.random((batch_size * 3, timesteps, data_dim))
# y_val = np.random.random((batch_size * 3, num_classes))
#
# model.fit(x_train, y_train,
#           batch_size=batch_size, epochs=5, shuffle=False,
#           validation_data=(x_val, y_val))
