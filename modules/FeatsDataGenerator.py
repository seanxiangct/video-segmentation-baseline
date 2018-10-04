import numpy as np
import cv2
from keras.applications.resnet50 import preprocess_input
from keras.utils import np_utils, Sequence
from skimage.io import imread


class FeatsDataGenerator(Sequence):
    """
    Generate features and label batches
    """

    def __init__(self, feats, labels, batch_size=32):
        """
        :param feats: data tuple (image path, label)
        :param labels: ground_truth labels
        :param batch_size: size of the generated batch
        """
        self.feats = feats
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        feats_batch = self.feats[idx * self.batch_size:(idx + 1) * self.batch_size]
        label_batch = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(feats_batch), np.array(label_batch)
