import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np
import matplotlib.pyplot as plt

from modules.metrics import ComputeMetrics
from keras.models import load_model

from modules.utils import read_features, mask_data, unmask

if __name__ == '__main__':
    """
    predict result with the given model name
    """

    batch_size = 10
    max_len = 6000
    n_classes = 7

    model_name = 'BiLSTM-500nodes-noMask-250attentionBefore-0.1l2-8.h5'
    remote_feats_path = '/home/cxia8134/dev/baseline/feats/'
    remote_model_path = '/home/cxia8134/dev/baseline/trained/'
    remote_temp_model_path = '/home/cxia8134/dev/baseline/temp'

    local_feats_path = '/Users/seanxiang/data/cholec80/feats/'
    local_model_path = '/Users/seanxiang/data/trained/'

    trial_metrics = ComputeMetrics(overlap=.1, bg_class=0)
    trial_metrics.set_classes(n_classes)

    X_test, Y_test = read_features(remote_feats_path, 'test')
    X_test_m, Y_test_, M_test = mask_data(X_test, Y_test, max_len, mask_value=-1)

    # model = load_model(remote_model_path + model_name)
    model = load_model('/home/cxia8134/dev/baseline/temp/BiLSTM-500nodes-noMask-LSTMattentionBefore-5-07-0.94.hdf5')

    y_pred = model.predict(
        x=X_test_m,
        batch_size=batch_size,
        verbose=1,
    )
    y_pred = unmask(y_pred, M_test)
    y_pred = np.array([y.argmax(1) for y in y_pred])

    Y_test = np.transpose(np.array([y.argmax(axis=1) for y in Y_test]))

    for i, p in enumerate(y_pred):
        split = 'video_{}'.format(i)

        y_true = Y_test[i]
        trial_metrics.add_predictions(split, p, y_true)

        np.savetxt('pred/{}.txt'.format(i), p, delimiter=',', fmt='%s')
        np.savetxt('pred/{}_true.txt'.format(i), Y_test[i], delimiter=',', fmt='%s')

    print()
    trial_metrics.print_scores()
    trial_metrics.print_trials()
    print()

