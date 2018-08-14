import numpy as np
from keras.utils import np_utils
from keras.layers import Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from utils import extract_frames, read_data, read_labels

model_name = 'baseline_1'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=8, min_lr=0.5e-6, mode='auto')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10)
tensor_board = TensorBoard('log/' + model_name)

# path to the directory storing data and labels
# at 1fps sample rate:
# data size: 14.4GB
# number of frames in total: 184578
local_train_path = '/Users/seanxiang/data/cholec80/train_frames/'
local_vali_path = '/Users/seanxiang/data/cholec80/vali_frames/'
local_label_path = '/Users/seanxiang/data/cholec80/phase_annotations/'

remote_train_path = '/home/cxia8134/data/train_frames/'
remote_vali_path = '/home/cxia8134/data/vali_frames/'
remote_label_path = '/home/cxia8134/data/phase_annotations/'

# extracting frames from videos at 1fps
fps = 25
# extract_frames(data_path, 25)

batch_size = 32
nb_classes = 7
nb_epoch = 200
input_rows, input_cols = 224, 224
input_channels = 3

# data_generator = ImageDataGenerator(#rescale=1./255,
#                                     validation_split=0.2,
#                                     preprocessing_function=preprocess_input)
# train_generator = data_generator.flow_from_directory(local_path + 'train_frames',
#                                                      # shuffle=False,
#                                                      target_size=(224, 224),
#                                                      batch_size=batch_size)
# test_generator = data_generator.flow_from_directory(local_path + 'vali_frames',
#                                                     target_size=(224, 224),
#                                                     batch_size=batch_size)

X_train = read_data(remote_train_path)
Y_train = read_labels(remote_label_path, fps, 1, 41)

X_vali = read_data(remote_vali_path)
Y_vali = read_labels(remote_label_path, fps, 42, 51)

# one-hot encoding
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_vali = np_utils.to_categorical(Y_vali, nb_classes)

# preprocess data in the same way as imagenet
X_train = preprocess_input(X_train, mode='tf')
X_vali = preprocess_input(X_vali, mode='tf')

# define model structure
# output feature vector of length 2048 for each frame
model = ResNet50(include_top=False,
                 weights='imagenet',
                 pooling='avg')

# adding classification layer
# for layer in model.layers:
#     layer.trainable = False
last = model.output
x = Dense(units=nb_classes, activation='softmax', name='fc1')(last)
fine_tuned_model = Model(model.input, x)

fine_tuned_model.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])
fine_tuned_model.summary()

# create custom generator
fine_tuned_model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_vali, Y_vali),
          callbacks=[lr_reducer, early_stopper, tensor_board])

fine_tuned_model.save('trained/' + model_name +'.h5')
