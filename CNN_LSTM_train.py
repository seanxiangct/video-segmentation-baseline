# fine-turned CNN with Bidirectional LSTM with single output
import numpy as np
from keras import Model, Input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers import LSTM, Bidirectional, TimeDistributed, Dense
from keras.models import load_model

from modules.metrics import ComputeMetrics
from modules.utils import read_features, read_from_file
from modules.DataGenerator import DataGenerator

if __name__ == '__main__':
    """
    Training CNN-BidirectionalLSTM in an end-to-end manner
    """

    batch_size = 8
    n_classes = 7
    n_nodes = 200
    n_timesteps = 30
    n_epoches = 200

    local_train_pair = '/Users/seanxiang/data/cholec80/train_labels/labels.txt'
    local_vali_pair = '/Users/seanxiang/data/cholec80/vali_labels/labels.txt'
    local_model_path = '/Users/seanxiang/data/trained/'

    remote_train_pair = '/home/cxia8134/data/old_labels/1-41.txt'
    remote_vali_pair = '/home/cxia8134/data/old_labels/41-51.txt'
    remote_model_path = '/home/cxia8134/dev/baseline/trained/'

    model_name = 'baseline_1.h5'
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6, mode='auto')
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10)
    tensor_board = TensorBoard('log/' + model_name)

    train_pair = read_from_file(local_train_pair)
    vali_pair = read_from_file(local_vali_pair)

    train_generator = DataGenerator(train_pair, n_classes, batch_size)
    vali_generator = DataGenerator(vali_pair, n_classes, batch_size)

    # defines model input
    # inputs = Input(shape=(batch_size, n_timesteps, n_nodes))

    # fine-turned ResNet50
    model = load_model(local_model_path + model_name)

    # remove the last softmax layer
    features_model = TimeDistributed(Model(inputs=model.input, outputs=model.layers[-2].output))

    # add LSTM layer with single output
    lstm_model = Bidirectional(LSTM(n_nodes,
                                    dropout=0.25,
                                    recurrent_dropout=0.25,
                                    return_sequences=False))(features_model.output)
    lstm_model = TimeDistributed(Dense(n_classes, activation="softmax"))(lstm_model)
    lstm_model.summary()

    lstm_model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

    lstm_model.fit_generator(generator=train_generator,
                             validation_data=vali_generator,
                             epochs=n_epoches,
                             verbose=1,
                             # class_weight=,
                             workers=6,
                             use_multiprocessing=True,
                             callbacks=[lr_reducer, early_stopper, tensor_board])
