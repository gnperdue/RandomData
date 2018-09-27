'''
split mnist train by target
'''
import numpy as np
import h5py
import os


def fill_hdf5(file_name, features, targets):
    if os.path.exists(file_name):
        os.remove(file_name)
    f = h5py.File(file_name, 'w')
    features_set = f.create_dataset(
        'features', np.shape(features), dtype='uint8', compression='gzip'
    )
    targets_set = f.create_dataset(
        'targets', np.shape(targets), dtype='uint8', compression='gzip'
    )
    features_set[...] = features
    targets_set[...] = targets
    f.close()


def test_hdf5(file_name, test_val):
    f = h5py.File(file_name, 'r')
    print('all {} - {}'.format(test_val, np.all(f['targets'][:] == test_val)))


f = h5py.File('mnist_train.hdf5', 'r')

for i in range(10):
    target_idx = (f['targets'][:] == i).reshape((-1,))
    new_features = np.compress(target_idx, f['features'], axis=0)
    new_targets = np.compress(target_idx, f['targets'], axis=0)
    out_name = 'mnist_train_%ds.hdf5' % i
    fill_hdf5(out_name, new_features, new_targets)
    test_hdf5(out_name, i)
