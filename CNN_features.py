import numpy as np
import matplotlib.pyplot as plt
from keras import Model

import h5py
import os
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
from utils import read_from_file, read_from_pair

if __name__ == '__main__':

    batch_size = 32
    nb_classes = 7

    local_model_path = '/Users/seanxiang/data/trained/baseline_1.h5'
    remote_model_path = '/home/cxia8134/dev/baseline/trained/baseline_1.h5'
    local_test_path = '/Users/seanxiang/data/cholec80/test_labels/labels.txt'
    local_predition_path = 'feats.txt'

    # path to validation data and labels
    # d_path = '/home/sean/Dev/data/vali-set/*.png'
    # l_path = '/home/sean/Dev/data/vali.txt'

    test_pair = read_from_file(local_test_path)
    X, y = read_from_pair(test_pair, nb_classes)

    model = load_model(local_model_path)

    features_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    features = features_model.predict(
        x=X,
        batch_size=batch_size,
        verbose=1
    )
    # y_pred = model.predict(
    #     x=X_test,
    #     batch_size=batch_size,
    #     verbose=1,
    # )
    y_pred = [np.argmax(i, axis=None, out=None) for i in y_pred]

    # calculating accuracy
    # acc = accuracy_score(y_test, y_pred)
    # print(acc)
    #
    # # draw confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # np.set_printoptions(precision=2)
    # plt.figure(figsize=(5.5, 4), dpi=300)
    # plot_confusion_matrix(cm, classes=classes, normalize=True)
    # plt.show()

    for i, filename in enumerate(sorted(os.listdir(t_path)), start=0):
        if filename.endswith(".png"):
            y_pred[i] = [filename, y_pred[i]]
    np.savetxt('../output/' + model_name, y_pred, delimiter=' ', fmt='%s')
