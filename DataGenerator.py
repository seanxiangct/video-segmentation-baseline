import numpy as np
from collections import Sequence
from keras.applications.resnet50 import preprocess_input
from skimage.io import imread


class DataGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        """

        :param image_filenames: an array of image file names
        :param labels: an array of corresponding labels
        :param batch_size: size of the generated batch
        """
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) + self.batch_size]

        return np.array([preprocess_input(imread(file_name)) for file_name in batch_x]), np.array(batch_y)
