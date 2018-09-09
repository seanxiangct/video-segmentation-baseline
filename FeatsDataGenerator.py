import numpy as np
import cv2
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils, Sequence
from skimage.io import imread


class FeatsDataGenerator(Sequence):

    def __init__(self, data_pair, nb_classes, batch_size=32):
        """
        :param image_filenames: an array of image file names
        :param labels: an array of corresponding labels
        :param batch_size: size of the generated batch
        """
        self.data_pair = data_pair
        self.batch_size = batch_size
        self.nb_classes = nb_classes

    def __len__(self):
        return int(np.ceil(len(self.data_pair) / float(self.batch_size)))

    def __getitem__(self, idx):
        pairs = self.data_pair[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_input = []
        batch_output = []
        for p in pairs:
            img_path = p[0]
            img = cv2.cvtColor(imread(img_path), cv2.COLOR_BGRA2BGR)
            img = preprocess_input(img, mode='tf')

            y = np_utils.to_categorical(int(p[1]), self.nb_classes)

            batch_input += [img]
            batch_output += [y]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        return batch_x, batch_y
