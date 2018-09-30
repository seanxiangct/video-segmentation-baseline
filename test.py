import os

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
import numpy as np
from keras.utils import np_utils

import utils.datasets
import utils.metrics
import utils
from TCN import TCN_LSTM, residual_TCN_LSTM
from utils.utils import read_from_file, read_features, mask_data

import seq2seq

local_feats_path = '/Users/seanxiang/data/cholec80/feats/'

remote_feats_path = '/home/cxia8134/dev/baseline/feats/'
remote_train_path = '/home/cxia8134/dev/baseline/feats/train'
remote_vali_path = '/home/cxia8134/dev/baseline/feats/vali'

model_name = 'test-attention-action-5'

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6, mode='auto')
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=10)
tensor_board = TensorBoard('log/' + model_name)

seq_length = 20
# n_nodes = [576, 288]
n_nodes = [128, 128]
nb_epoch = 200
nb_classes = 18
batch_size = 32
conv_len = [8, 16, 32, 64, 128][2]
n_feat = 2048
max_len = 6000

# X_train, Y_train = read_features(local_feats_path, 'train')
# X_vali, Y_vali = read_features(local_feats_path, 'vali')
#
# X_train_m, Y_train_, M_train = mask_data(X_train, Y_train, max_len, mask_value=-1)
# X_vali_m, Y_vali_, M_vali = mask_data(X_vali, Y_vali, max_len, mask_value=-1)

local_path = "Dropbox/2018_yr4/Honour/dev/TemporalConvolutionalNetworks/"
remote_path = "dev/TemporalConvolutionalNetworks/"
base_dir = os.path.expanduser("~/{}".format(remote_path))

save_predictions = [False, True][0]
viz_predictions = [False, True][0]
viz_weights = [False, True][0]

# Set dataset and action label granularity (if applicable)
dataset = ["50Salads", "JIGSAWS", "MERL", "GTEA"][0]
granularity = ["eval", "mid"][1]
sensor_type = ["video", "sensors"][0]

# Set model and parameters
model_type = ["SVM", "LSTM", "LC-SC-CRF", "tCNN", "DilatedTCN", "ED-TCN", "TDNN"][1]
# causal or acausal? (If acausal use Bidirectional LSTM)

# How many latent states/nodes per layer of network
# Only applicable to the TCNs. The ECCV and LSTM  model suses the first element from this list.
# n_nodes = [64, 96]
video_rate = 3

# Which features for the given dataset
features = "SpatialCNN"
bg_class = 0 if dataset is not "JIGSAWS" else None

if dataset == "50Salads":
    features = "SpatialCNN_" + granularity

data = datasets.Dataset(dataset, base_dir)
trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=bg_class)

for i, split in enumerate(data.splits):
    if i == 0:
        if sensor_type == "video":
            feature_type = "A" if model_type != "SVM" else "X"
        else:
            feature_type = "S"

        X_train, y_train, X_test, y_test = data.load_split(features, split=split,
                                                           sample_rate=video_rate,
                                                           feature_type=feature_type)

        if trial_metrics.n_classes is None:
            trial_metrics.set_classes(data.n_classes)

        n_classes = data.n_classes

        # the length of each video
        train_lengths = [x.shape[0] for x in X_train]
        test_lengths = [x.shape[0] for x in X_test]
        n_train = len(X_train)
        n_test = len(X_test)

        n_feat = data.n_features
        print("# Feat:", n_feat)

        Y_train = [np_utils.to_categorical(y, n_classes) for y in y_train]
        Y_test = [np_utils.to_categorical(y, n_classes) for y in y_test]

        # In order process batches simultaneously all data needs to be of the same length
        # So make all same length and mask out the ends of each.
        n_layers = len(n_nodes)
        max_len = max(np.max(train_lengths), np.max(test_lengths))
        max_len = int(np.ceil(max_len / (2 ** n_layers))) * 2 ** n_layers
        print("Max length:", max_len)

        X_train_m, Y_train_, M_train = utils.mask_data(X_train, Y_train, max_len, mask_value=-1)
        X_test_m, Y_test_, M_test = utils.mask_data(X_test, Y_test, max_len, mask_value=-1)

        # ED-CNN
        # model = TCN_LSTM(n_nodes=n_nodes,
        #                  conv_len=conv_len,
        #                  n_classes=n_classes,
        #                  n_feat=n_feat,
        #                  max_len=max_len,
        #                  online=False,
        #                  activation='norm_relu',
        #                  return_param_str=False)
        model = seq2seq.AttentionSeq2Seq(input_dim=n_feat,
                                         input_length=max_len,
                                         hidden_dim=50,
                                         output_dim=n_classes,
                                         output_length=max_len,
                                         depth=3)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      sample_weight_mode="temporal",
                      metrics=['accuracy'])

        # train with extracted features from each video
        model.fit(x=X_train_m,
                  y=Y_train_,
                  epochs=nb_epoch,
                  batch_size=16,
                  verbose=1,
                  validation_split=0.1,
                  sample_weight=M_train[:, :, 0],
                  shuffle=False,
                  callbacks=[lr_reducer, early_stopper, tensor_board])

        AP_train = model.predict(X_train_m, verbose=0)
        AP_test = model.predict(X_test_m, verbose=0)
        AP_train = utils.unmask(AP_train, M_train)
        AP_test = utils.unmask(AP_test, M_test)

        P_train = [p.argmax(1) for p in AP_train]
        P_test = [p.argmax(1) for p in AP_test]

        trial_metrics.add_predictions(split, P_test, y_test)
        trial_metrics.print_trials()
        print()

        print()
        trial_metrics.print_scores()
        trial_metrics.print_trials()
        print()
