import numpy as np
import csv
import matplotlib.pyplot as plt
from skvideo.io import vreader
from skimage.transform import resize
from skimage.io import imread_collection

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
                frame = resize(frame, (224, 224, 3), anti_aliasing=True)
                # extract frame to different folder
                frame_path = '{}frames/video{}-frame{}.png'.format(d_path, video_id, i)
                plt.imsave(arr=frame, fname=frame_path)


def read_data(d_path):
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

