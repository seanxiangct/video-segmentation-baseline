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
from keras.preprocessing.image import load_img

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
    data_path = '{}frames/'.format(path, source)
    for n in range(start, end):
        video_id = str(n).zfill(2)
        label_path = '{}phase_annotations/video{}-phase.txt'.format(path, video_id)
        dest_path = '{}{}_labels/labels.txt'.format(path, source)
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
            img = preprocess_input(img)

            y = np_utils.to_categorical(int(pair[i][1]), nb_classes)

            batch_input += [img]
            batch_output += [y]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield (batch_x, batch_y)
