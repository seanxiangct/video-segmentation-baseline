import h5py

import numpy as np
import csv
import matplotlib.pyplot as plt
import glob
import os
import cv2
from keras.utils import np_utils
from skvideo.io import vreader
from skimage.transform import resize
from skimage.io import imread_collection, imread
from keras.applications.resnet50 import preprocess_input

phase_mapping = {
    'Preparation': 0,
    'CalotTriangleDissection': 1,
    'ClippingCutting': 2,
    'GallbladderDissection': 3,
    'GallbladderPackaging': 4,
    'CleaningCoagulation': 5,
    'GallbladderRetraction': 6
}


def extract_frames(d_path, fps=25):
    """
    extract frames from video files, 1 frame per second by default
    :param d_path: data path
    :param fps: number of frames being extracted per second
    :return: data and labels
    """
    for n in range(1, 81):
        video_id = str(n).zfill(2)
        video_name = '{}videos/video{}.mp4'.format(d_path, video_id)
        video_reader = vreader(video_name)
        for i, frame in enumerate(video_reader):
            if i % fps == 0:
                # remove black boarders
                frame = frame[:, 40:-55, :]
                # downscale frame to (224, 224, 3)
                frame = resize(frame, (224, 224, 3))
                # extract frame to different folder
                frame_path = '{}frames/video{}-frame{}.png'.format(d_path, video_id, i)
                plt.imsave(arr=frame, fname=frame_path)


def read_frames(d_path):
    ic = imread_collection(d_path + '*.png')
    data = []
    for image in ic:
        data.append(np.asarray(image / 255.0, dtype=np.float64))
    data = np.asarray(data)

    return data


def read_labels(d_path, fps, start, end):
    """
    read labels from directory
    :param d_path: path to the directory containing all the label files
    :param fps: sample rate
    :param start: starting file name
    :param end: ending file name
    :return: labels
    """
    y = []

    for n in range(start, end):
        video_id = str(n).zfill(2)
        label_path = '{}video{}-phase.txt'.format(d_path, video_id)
        with open(label_path) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for i, row in enumerate(reader):
                if i % fps == 0:
                    phase_name = row[1]
                    phase_id = phase_mapping[phase_name]
                    y.append(phase_id)

    y = np.asarray(y)

    return y


def read_data(path, fps, start, end, source):
    """
    raed both frame paths and their labels
    :return: X and Y
    """
    # read image names
    f_path = path + 'frames'
    f_names = sorted(glob.glob(f_path), key=os.path.getmtime)
    # if source == 'train':
    #     f_path += 'train_frames/*.png'
    # elif source == 'vali':
    #     f_path += 'vali_frames/*.png'

    y = []
    y_path = path + 'phase_annotations/'
    for n in range(start, end):
        video_id = str(n).zfill(2)
        label_path = '{}video{}-phase.txt'.format(y_path, video_id)
        with open(label_path) as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for i, row in enumerate(reader):
                if i % fps == 0:
                    phase_name = row[1]
                    phase_id = phase_mapping[phase_name]
                    y.append(phase_id)

    y = np.asarray(y)

    return f_names, y


def read_labels_to_file(path, fps, start, end, source):
    """
    Givne start and end index of videos, save (frame path, label) pairs into file
    :param path:
    :param fps:
    :param start:
    :param end:
    :param source:
    :return:
    """
    data_path = '{}frames/'.format(path)
    for n in range(start, end):
        video_id = str(n).zfill(2)
        label_path = '{}phase_annotations/video{}-phase.txt'.format(path, video_id)
        dest_path = '{}{}_labels/{}-{}.txt'.format(path, source, start, end)
        with open(dest_path, 'a') as new_f:
            writer = csv.writer(new_f, delimiter='\t')
            with open(label_path) as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)
                for i, row in enumerate(reader):
                    if i % fps == 0:
                        phase_name = row[1]
                        phase_id = phase_mapping[phase_name]
                        img_path = '{}video{}-frame{}.png'.format(data_path, video_id, i)
                        writer.writerow([img_path, phase_id])


def data_generator(path, fps, start, end, source='train_frames', batch_size=32):
    data_path = '{}{}/'.format(path, source)
    label_path = path + 'phase_annotations/'
    while 1:
        for n in range(start, end):
            video_id = str(n).zfill(2)
            label_path = '{}video{}-phase.txt'.format(label_path, video_id)
            with open(label_path) as f:
                reader = csv.reader(f, delimiter='\t')
                next(reader)
                for i, row in enumerate(reader):
                    if i % fps == 0:
                        phase_name = row[1]
                        phase_id = phase_mapping[phase_name]
                        img = imread('{}video{}-frame{}.png'.format(data_path, video_id, i))
                        yield (preprocess_input(img), phase_id)
                f.close()


def read_from_file(path):
    pair = []
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            pair.append(row)
    return pair


def data_generator_from_labels(pair, idx, nb_classes, batch_size=32):
    while 1:
        rand_indx = np.random.choice(a=idx, size=batch_size, replace=False)

        batch_input = []
        batch_output = []
        for i in rand_indx:
            img_path = pair[i][0]
            img = cv2.cvtColor(imread(img_path), cv2.COLOR_BGRA2BGR)
            img = preprocess_input(img, mode='tf')

            y = np_utils.to_categorical(int(pair[i][1]), nb_classes)

            batch_input += [img]
            batch_output += [y]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)


def shuffle(train, label):
    """
    shuffle the training data and labels together
    """
    idx = np.random.permutation(train.shape[0])
    return train[idx], label[idx]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.rcParams.update({'font.size': 3})
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=3)
    plt.yticks(tick_marks, classes, fontsize=3)

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def read_from_pair(pairs, nb_classes):
    """
    read from x-x+1.txt file, load frames and one-hot-encode their labels
    :param pairs:
    :param nb_classes:
    :return: preprocessed frames and one-hot-encoded labels
    """
    data = []
    labels = []
    for pair in pairs:
        img = cv2.cvtColor(imread(pair[0]), cv2.COLOR_BGRA2BGR)
        img = preprocess_input(img, mode='tf')
        data.append(img)

        labels.append(pair[1])

    labels = np_utils.to_categorical(labels, nb_classes)
    return np.array(data), np.array(labels)


def read_features(path, source):
    """
    read features, labels and length from x-x+1.h5 feature files
    :param path: path of the feats folder
    :param source: subdirectory, could be train, vali or test
    :return: features, one-hot-encoded labels, and video length
    """

    train = []
    labels = []
    feats_path = path + source
    for i, f in enumerate(sorted(os.listdir(feats_path)), start=0):
        h5f = h5py.File('{}/{}'.format(feats_path, f), 'r')
        train.append(h5f['feats'][:])
        labels.append(h5f['y'][:])
        h5f.close()
    return train, labels


def mask_data(X, Y, max_len=None, mask_value=0):
    """Mask data and labels, padding is added to the end of the sequence"""
    if max_len is None:
        max_len = np.max([x.shape[0] for x in X])
    X_ = np.zeros([len(X), max_len, X[0].shape[1]]) + mask_value
    Y_ = np.zeros([len(X), max_len, Y[0].shape[1]]) + mask_value
    mask = np.zeros([len(X), max_len])
    for i in range(len(X)):
        l = X[i].shape[0]
        X_[i, :l] = X[i]
        Y_[i, :l] = Y[i]
        mask[i, :l] = 1
    return X_, Y_, mask[:, :, None]


def unmask(X, M):
    if X[0].ndim == 1 or (X[0].shape[0] > X[0].shape[1]):
        return [X[i][M[i].flatten() > 0] for i in range(len(X))]
    else:
        return [X[i][:, M[i].flatten() > 0] for i in range(len(X))]


# ----------------- test functions ---------------

def remap_labels(Y_all):
    # Map arbitrary set of labels (e.g. {1,3,5}) to contiguous sequence (e.g. {0,1,2})
    ys = np.unique([np.hstack([np.unique(Y_all[i]) for i in range(len(Y_all))])])
    y_max = ys.max()
    y_map = np.zeros(y_max + 1, np.int) - 1
    for i, yi in enumerate(ys):
        y_map[yi] = i
    Y_all = [y_map[Y_all[i]] for i in range(len(Y_all))]
    return Y_all


def subsample(X, Y, rate=1, dim=0):
    if dim == 0:
        X_ = [x[::rate] for x in X]
        Y_ = [y[::rate] for y in Y]
    elif dim == 1:
        X_ = [x[:, ::rate] for x in X]
        Y_ = [y[::rate] for y in Y]
    else:
        print("Subsample not defined for dim={}".format(dim))
        return None, None

    return X_, Y_


# ------------- Segment functions -------------
def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Yi_split


def segment_data(Xi, Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    Xi_split = [np.squeeze(Xi[:, idxs[i]:idxs[i + 1]]) for i in range(len(idxs) - 1)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs) - 1)])
    return Xi_split, Yi_split


def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
    return intervals


def segment_lengths(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0] + 1).tolist() + [len(Yi)]
    intervals = [(idxs[i + 1] - idxs[i]) for i in range(len(idxs) - 1)]
    return np.array(intervals)

####
