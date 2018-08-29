import numpy as np
from DataGenerator import DataGenerator
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from utils import extract_frames, read_data, read_labels, data_generator_from_labels, read_from_file

if __name__ == '__main__':

    model_name = 'baseline_3_ordered_sequence_tfmode'

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=8, min_lr=0.5e-6, mode='auto')
    early_stopper = EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=10)
    tensor_board = TensorBoard('log/' + model_name)

    # path to the directory storing data and labels
    # at 1fps sample rate:
    # data size: 14.4GB
    # number of frames in total: 184578
    local_path = '/Users/seanxiang/data/cholec80/'
    local_train_path = '/Users/seanxiang/data/cholec80/train_frames/'
    local_vali_path = '/Users/seanxiang/data/cholec80/vali_frames/'
    local_label_path = '/Users/seanxiang/data/cholec80/phase_annotations/'
    local_train_pair = '/Users/seanxiang/data/cholec80/train_labels/labels.txt'
    local_vali_pair = '/Users/seanxiang/data/cholec80/vali_labels/labels.txt'

    remote_path = '/home/cxia8134/data/'
    remote_train_path = '/home/cxia8134/data/train_frames/'
    remote_vali_path = '/home/cxia8134/data/vali_frames/'
    remote_label_path = '/home/cxia8134/data/phase_annotations/'
    remote_train_pair = '/home/cxia8134/data/train_labels/labels.txt'
    remote_vali_pair = '/home/cxia8134/data/vali_labels/labels.txt'

    train_folder = 'train_frames'
    vali_folder = 'vali_frames'

    # extracting frames from videos at 1fps
    fps = 25
    # extract_frames(data_path, 25)

    batch_size = 32
    nb_classes = 7
    nb_epoch = 200
    input_height, input_width = 224, 224
    input_channels = 3
    n_train = 86344
    n_vali = 21108

    # data_generator = ImageDataGenerator(validation_split=0.2,
    #                                     preprocessing_function=preprocess_input)
    # train_generator = data_generator.flow_from_directory(remote_train_path + 'train_frames',
    #                                                      # shuffle=False,
    #                                                      target_size=(224, 224),
    #                                                      batch_size=batch_size)
    # vali_generator = data_generator.flow_from_directory(remote_vali_path + 'vali_frames',
    #                                                     target_size=(224, 224),
    #                                                     batch_size=batch_size)

    # X_train = read_data(remote_train_path)
    # Y_train = read_labels(remote_label_path, fps, 1, 41)
    #
    # X_vali = read_data(remote_vali_path)
    # Y_vali = read_labels(remote_label_path, fps, 42, 51)

    # one-hot encoding
    # Y_train = np_utils.to_categorical(Y_train, nb_classes)
    # Y_vali = np_utils.to_categorical(Y_vali, nb_classes)

    # preprocess data in the same way as imagenet
    # X_train = preprocess_input(X_train, mode='tf')
    # X_vali = preprocess_input(X_vali, mode='tf')


    # train_pair = read_from_file(local_train_pair)
    # vali_pair = read_from_file(local_vali_pair)
    train_pair = read_from_file(remote_train_pair)
    vali_pair = read_from_file(remote_vali_pair)

    # testx, testy = data_generator_test(train_pair, nb_classes, batch_size)

    # unordered data generator
    # train_idx = np.array(len(train_pair))
    # vali_idx = np.array(len(vali_pair))
    # train_generator = data_generator_from_labels(train_pair, train_idx, nb_classes, batch_size)
    # vali_generator = data_generator_from_labels(vali_pair, vali_idx, nb_classes, batch_size)

    # ordered data generator
    train_generator = DataGenerator(train_pair, nb_classes, batch_size)
    vali_generator = DataGenerator(vali_pair, nb_classes, batch_size)

    # X_train, Y_train = read_data(local_path, fps, 1, 41, 'train')
    # X_vali, Y_vali = read_data(local_path, fps, 42, 51, 'vali')
    #
    # train_generator = DataGenerator(X_train, Y_train, batch_size)
    # vali_generator = DataGenerator(X_vali, Y_vali, batch_size)

    # define model structure
    # output feature vector of length 2048 for each frame
    model = ResNet50(include_top=False,
                     weights='imagenet',
                     input_shape=(input_height, input_width, input_channels),
                     pooling='avg')

    # adding classification layer
    last = model.output
    x = Dense(units=nb_classes, activation='softmax', name='fc1')(last)

    fine_tuned_model = Model(model.input, x)

    fine_tuned_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
    fine_tuned_model.summary()

    # create custom generator
    fine_tuned_model.fit_generator(generator=train_generator,
                                   steps_per_epoch=(n_train // batch_size),
                                   epochs=nb_epoch,
                                   validation_data=vali_generator,
                                   validation_steps=(n_vali // batch_size),
                                   verbose=1,
                                   use_multiprocessing=True,
                                   workers=6,
                                   max_queue_size=16,
                                   callbacks=[lr_reducer, early_stopper, tensor_board])

    fine_tuned_model.save('trained/' + model_name + '.h5')
