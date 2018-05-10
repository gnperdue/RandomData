from __future__ import print_function
from __future__ import absolute_import

import os
import tensorflow as tf

from read_hdf5_mnist import HDF5Reader

# Get path to data
DDIR = os.environ['HOME'] + '/Dropbox/Data/RandomData/hdf5'
TFILE = DDIR + '/mnist_test.hdf5'


def make_batch_generator_fn(hdf5_file_name, batch_size):
    """
    make a generator function that we can query for batches
    """
    reader = HDF5Reader(hdf5_file_name, batch_size)
    nevents = reader.openf()

    def batch_generator_fn():
        start_idx = 0
        stop_idx = batch_size
        while True:
            # endlessly supply batches until we call `close()`
            if stop_idx > nevents:
                return
            yield reader.get_features(start_idx, stop_idx), \
                reader.get_targets(start_idx, stop_idx)
            start_idx += batch_size
            stop_idx += batch_size
            
        # close the reader when we leave the batch generation
        reader.closef()

    return batch_generator_fn


def one_shot_iterator_read(batch_size=25):
    # make a generator function
    dgen = make_batch_generator_fn(TFILE, batch_size)

    # make a Dataset from a generator
    features_shape = [None, 1, 28, 28]
    targets_shape = [None, 1]
    ds = tf.data.Dataset.from_generator(
        dgen, (tf.uint8, tf.uint8),
        (tf.TensorShape(features_shape), tf.TensorShape(targets_shape))
    )
    # one_shot_iterators do not have initializers
    itrtr = ds.make_one_shot_iterator()
    feats, targs = itrtr.get_next()

    # sess
    with tf.Session() as sess:
        counter = 0
        try:
            while True:
                f, t = sess.run([feats, targs])
                # print(f.shape, f.dtype, t.shape, t.dtype)
                print(t.reshape(-1,))
                counter += 1
                if counter > 1100:
                    break
        except IndexError:
            print('end of dataset at counter = {}'.format(counter))
        except tf.errors.OutOfRangeError:
            print('end of dataset at counter = {}'.format(counter))
        except Exception as e:
            print(e)


def simple_read(batch_size=25):
    # make a generator function
    dgen = make_batch_generator_fn(TFILE, batch_size)

    count = 0

    for f, t in dgen():
        count += 1
        if count > 10:
            break
        print(f.shape, f.dtype, t.shape, t.dtype)

        f_temp, t_temp = next(dgen())

        # close the generator?


if __name__ == '__main__':
    one_shot_iterator_read()
