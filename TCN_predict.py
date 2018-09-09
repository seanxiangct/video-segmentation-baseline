import numpy as np
import matplotlib.pyplot as plt

import metrics
import resnet
import csv
import os
from keras.models import load_model

from utils import read_features, mask_data, unmask

if __name__ == '__main__':

    batch_size = 32
    max_len = 6000
    n_classes = 7

    model_name = 'TCN-LSTM-96,192nodes-.h5'
    remote_feats_path = '/home/cxia8134/dev/baseline/feats/'
    remote_model_path = '/home/cxia8134/dev/baseline/trained/'

    local_feats_path = '/Users/seanxiang/data/cholec80/feats/'
    local_model_path = 'Users/seanxiang/data/trained/'

    trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=0)
    trial_metrics.set_classes(n_classes)

    X_test, Y_test = read_features(local_feats_path, 'test')
    X_test_m, Y_test_, M_test = mask_data(X_test, Y_test, max_len, mask_value=-1)

    model = load_model(local_model_path + model_name)

    y_pred = model.predict(
        x=X_test_m,
        batch_size=batch_size,
        verbose=1,
    )
    y_pred = unmask(y_pred, M_test)
    y_pred = [y.argmax(1) for y in y_pred]

    for i, p in enumerate(y_pred):
        split = 'video_{}'.format(i)
        trial_metrics.add_predictions(split, y_pred, Y_test)
        trial_metrics.print_trials()
        print()

        print()
        trial_metrics.print_scores()
        trial_metrics.print_trials()
        print()

        np.savetxt('pred/{}'.format(i), p, delimiter=',', fmt='%s')
        np.savetxt('pred/{}_true'.format(i), Y_test[i], delimiter=',', fmt='%s')

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
