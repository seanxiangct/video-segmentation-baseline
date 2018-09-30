from fnmatch import fnmatch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_similarity_score

import utils.metrics
import resnet
import csv
import os
from keras.models import load_model

from utils.utils import read_features, mask_data, unmask, read_from_file, read_from_pair


if __name__ == '__main__':
    batch_size = 32
    nb_classes = 7

    local_model_path = '/Users/seanxiang/data/trained/baseline_1.h5'
    local_test_path = '/Users/seanxiang/data/cholec80/test_labels'

    remote_model_path = '/home/cxia8134/dev/baseline/trained/baseline_4_unordered_tfmode.h5'
    remote_train_path = '/home/cxia8134/data/train_labels'
    remote_vali_path = '/home/cxia8134/data/vali_labels'
    remote_test_path = '/home/cxia8134/data/test_labels'

    model = load_model(remote_model_path)
    trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=0)
    trial_metrics.set_classes(nb_classes)

    acc = []
    for i, f in enumerate(sorted(os.listdir(remote_test_path)), start=0):

        if fnmatch(f, '*.txt'):
            print('predicting {}'.format(f))
            test_pair = read_from_file('{}/{}'.format(remote_test_path, f))
            X, y = read_from_pair(test_pair, nb_classes)

            y_pred = model.predict(
                x=X,
                batch_size=batch_size,
                verbose=1,
            )

            y_pred = np.array([y.argmax(axis=0) for y in y_pred])
            y_true = np.transpose(np.array([y.argmax(axis=0) for y in y]))

            trial_metrics.add_predictions('video_{}'.format(i), y_pred, y_true)
            trial_metrics.print_trials()
            print()

            print()
            trial_metrics.print_scores()
            trial_metrics.print_trials()
            print()

            print('Jaccard score for video_{} is {}'.format(i, jaccard_similarity_score(y_true=y_true,
                                                                                        y_pred=y_pred,
                                                                                        normalize=True)))

            # np.savetxt('pred/cnn/{}'.format(f), y_pred, delimiter=',', fmt='%s')

    # y_pred = [np.argmax(i, axis=None, out=None) for i in y_pred]

    # calculating accuracy
    # acc = accuracy_score(y_test, y_pred)
    # print(acc)
    #
    # # draw confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # np.set_printoptions(precision=2)
    # classes = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E',
    # 'F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W',
    # 'X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
    # 'p','q','l','s','t','u','v','w','x','y','z']
    #
    # plt.figure(figsize=(5.5, 4), dpi=300)
    # plot_confusion_matrix(cm, classes=classes, normalize=True)
    # plt.show()

    # for i, filename in enumerate(sorted(os.listdir(t_path)), start=0):
    #     if filename.endswith(".png"):
    #         y_pred[i] = [filename, y_pred[i]]
    # np.savetxt('../output/' + model_name, y_pred, delimiter=' ', fmt='%s')
