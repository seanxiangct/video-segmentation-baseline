import h5py
import os
from keras.models import load_model
from keras import Model
from fnmatch import fnmatch
from utils.utils import read_from_file, read_from_pair

def extract_features(model, path, source, batch_size=32, nb_classes=7):
    max_len = 0
    for i, f in enumerate(sorted(os.listdir(path)), start=0):

        if fnmatch(f, '*.txt'):
            print('extracting features for {}'.format(f))
            test_pair = read_from_file('{}/{}'.format(path, f))
            X, y = read_from_pair(test_pair, nb_classes)

            features = model.predict(
                x=X,
                batch_size=batch_size,
                verbose=1
            )
            print(features.shape[0])
            if features.shape[0] > max_len:
                max_len = features.shape[0]

            features_file = h5py.File('feats/{}/{}.h5'.format(source, f[:-4]), 'w')
            features_file.create_dataset('feats', data=features)
            features_file.create_dataset('length', features.shape)
            features_file.create_dataset('y', data=y)
            features_file.close()
    print(max_len)


if __name__ == '__main__':

    batch_size = 32
    nb_classes = 7

    local_model_path = '/Users/seanxiang/data/trained/baseline_4_unordered_tfmode.h5'
    local_test_path = '/Users/seanxiang/data/cholec80/test_labels'

    remote_model_path = '/home/cxia8134/dev/baseline/trained/baseline_4_unordered_tfmode.h5'
    remote_train_path = '/home/cxia8134/data/train_labels'
    remote_vali_path = '/home/cxia8134/data/vali_labels'
    remote_test_path = '/home/cxia8134/data/test_labels'

    model = load_model(remote_model_path)
    features_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    extract_features(features_model, remote_train_path, 'train')
    extract_features(features_model, remote_vali_path, 'vali')
    extract_features(features_model, remote_test_path, 'test')
