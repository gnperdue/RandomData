'''
convert the keras (numpy) fashion mnist dataset to hdf5 for file pipeline work
'''
import numpy as np
import h5py
import os
from tensorflow import keras


def fill_hdf5(file_name, images, labels):
    f = h5py.File(file_name, 'w')
    grp = f.create_group('fashion')
    images_set = grp.create_dataset(
        'images', np.shape(images), dtype='uint8', compression='gzip'
    )
    labels_set = grp.create_dataset(
        'labels', np.shape(labels), dtype='uint8', compression='gzip'
    )
    images_set[...] = images
    labels_set[...] = labels
    f.close()


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = \
    fashion_mnist.load_data()

test_labels = np.expand_dims(test_labels, -1)
test_images = np.expand_dims(test_images, -1)
train_labels = np.expand_dims(train_labels, -1)
train_images = np.expand_dims(train_images, -1)

hdf5_train = 'fashion_train.hdf5'
hdf5_test = 'fashion_test.hdf5'
for f in [hdf5_train, hdf5_test]:
    if os.path.exists(f):
        os.remove(f)

fill_hdf5(hdf5_train, train_images, train_labels)
fill_hdf5(hdf5_test, test_images, test_labels)
