"""
convert the hdf5 mnist file to tfrecords
"""
import h5py
import tensorflow as tf
import numpy as np
import os
import gzip
import shutil
import matplotlib.pyplot as plt


class FashionHDF5Reader(object):
    """
    user should call `openf()` and `closef()` to start/finish.

    assumes stored image shape is [N, depth(=1 greyscale), H, W]
    """

    def __init__(self, hdf5_file):
        self._file = hdf5_file
        self._f = None
        self._nlabels = 10
        self.class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['fashion/labels'].shape[0]
        return self._nevents

    def closef(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_example(self, idx):
        image = self._f['fashion/images'][idx]
        image = np.moveaxis(image, 0, -1)
        label = self._f['fashion/labels'][idx].reshape([-1])
        return image, label

    def get_flat_example(self, idx):
        image, label = self.get_example(idx)
        image = np.reshape(image, (28 * 28))
        return image, label


def gz_compress(infile):
    outfile = infile + '.gz'
    with open(infile, 'rb') as f_in, gzip.open(outfile, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    if os.path.isfile(outfile) and (os.stat(outfile).st_size > 0):
        os.remove(infile)
    else:
        raise IOError('Compressed file not produced!')


def get_binary_data(reader, idx):
    """
    * reader - mnist_hdf5_reader
    * index
    returns a tuple of byte data (features, targets)
    """
    tfeat, ttarg = reader.get_flat_example(idx)
    return tfeat.tobytes(), ttarg.tobytes()


def write_tfrecord(reader, start_idx, stop_idx, tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for idx in range(start_idx, stop_idx):
        tfeat, ttarg = get_binary_data(reader, idx)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'images': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tfeat])
                    ),
                    'labels': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[ttarg])
                    )
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()


def tfrecord_to_graph_ops(filenames, num_epochs):
    file_queue = tf.train.string_input_producer(
        filenames, name='file_queue', num_epochs=num_epochs
    )
    reader = tf.TFRecordReader(
        options=tf.python_io.TFRecordOptions(
            compression_type=tf.python_io.TFRecordCompressionType.GZIP
        )
    )
    _, tfrecord = reader.read(file_queue)

    tfrecord_features = tf.parse_single_example(
        tfrecord,
        features={
            'images': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
        },
        name='data'
    )
    tfeat = tf.decode_raw(tfrecord_features['images'], tf.uint8)
    # note, 'NCHW' is only supported on GPUs, so use 'NHWC'...
    tfeat = tf.reshape(tfeat, [-1, 28, 28, 1])
    ttarg = tf.decode_raw(tfrecord_features['labels'], tf.uint8)
    ttarg = tf.one_hot(indices=ttarg, depth=10, on_value=1, off_value=0)
    return tfeat, ttarg


def batch_generator(filenames, batch_size=1, num_epochs=1):
    tfeat, ttarg = tfrecord_to_graph_ops(filenames, num_epochs)
    capacity = 10 * batch_size
    feats, targs = tf.train.batch(
        [tfeat, ttarg],
        batch_size=batch_size,
        capacity=capacity,
        enqueue_many=True,
        allow_smaller_final_batch=True
    )
    return feats, targs


def test_read_tfrecord(tfrecord_file):
    tfeat, ttarg = batch_generator([tfrecord_file])
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            for _ in range(10):
                features, targets = sess.run([tfeat, ttarg])
                print(features.shape)
                print(targets.shape)
                print(tf.argmax(targets, axis=1).eval())
                plt.imshow(features[0, :, :, 0])
                plt.show()
        except tf.errors.OutOfRangeError:
            print('out of examples')
        except Exception as e:
            print(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def write_all():
    m_train = FashionHDF5Reader('fashion_train.hdf5')
    n = m_train.openf()
    if not os.path.isfile('fashion_train.tfrecord.gz'):
        write_tfrecord(m_train, 0, n, 'fashion_train.tfrecord')
        gz_compress('fashion_train.tfrecord')
    m_train.closef()
    m_test = FashionHDF5Reader('fashion_test.hdf5')
    n = m_test.openf()
    if not os.path.isfile('fashion_test.tfrecord' + '.gz'):
        write_tfrecord(m_test, 0, n, 'fashion_test.tfrecord')
        gz_compress('fashion_test.tfrecord')
    m_test.closef()


def read_all():
    test_read_tfrecord('fashion_train.tfrecord.gz')
    test_read_tfrecord('fashion_test.tfrecord.gz')


if __name__ == '__main__':
    write_all()
    read_all()
