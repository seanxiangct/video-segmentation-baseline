from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import numpy as np

from modules.utils import read_features, mask_data
from modules.FeatsDataGenerator import FeatsDataGenerator
from Boundary_detection import boundary_sensitive_TCN


local_feats_path = '/Users/seanxiang/data/cholec80/feats/'
remote_feats_path = '/home/cxia8134/dev/baseline/feats/'

model_name = 'Shot_detection'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=7, min_lr=0.5e-6, mode='auto')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
tensor_board = TensorBoard('log/' + model_name)

seq_length = 20
n_nodes = [64, 128, 256]
nb_epoch = 200
nb_classes = 7
batch_size = 32
conv_len = [8, 16, 32, 64, 128][2]
n_feat = 2048
max_len = 6000

k = 250

X_train, Y_train = read_features(local_feats_path, 'train')
X_vali, Y_vali = read_features(local_feats_path, 'vali')

train_generator = FeatsDataGenerator(X_train, Y_train, batch_size)
vali_generator = FeatsDataGenerator(X_vali, Y_vali, batch_size)

model = boundary_sensitive_TCN(k=k)

model.fit_generator(generator=train_generator,
                    epochs=nb_epoch,
                    verbose=1,
                    validation_data=vali_generator,
                    callbacks=[lr_reducer, early_stopper, tensor_board])

# model predict
# argmax over the result to find the shot boundary frames within each run
# batch features based on the shot boundaries

model.predict()
