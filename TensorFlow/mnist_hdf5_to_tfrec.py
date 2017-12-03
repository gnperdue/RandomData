"""
convert the hdf5 mnist file to tfrecords
"""
import h5py
import tensorflow as tf
import os
import gzip
import shutil


class mnist_hdf5_reader:
    """
    mnist hdf5 files has `features` and `targets` w/ shapes (70000, 1, 28, 28)
    and (70000, 1) and dtype (both) uint8. the `mnist_hdf5_reader` will return
    numpy ndarrays of data for given ranges. user should call `open()` and
    `close()` to start/finish.
    """
    def __init__(self, hdf5_file):
        self.file = hdf5_file
        self._f = None

    def open(self):
        self._f = h5py.File(self.file, 'r')

    def close(self):
        try:
            self._f.close()
        except AttributeError:
            print('hdf5 file is not open yet.')

    def get_features(self, start_idx, stop_idx):
        return self._f['features'][start_idx: stop_idx]

    def get_targets(self, start_idx, stop_idx):
        return self._f['targets'][start_idx: stop_idx]

    def get_feature(self, idx):
        return self._f['features'][idx]

    def get_target(self, idx):
        return self._f['targets'][idx]


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
    tfeat = reader.get_feature(idx)
    ttarg = reader.get_target(idx)
    return tfeat.tobytes(), ttarg.tobytes()


def write_tfrecord(reader, start_idx, stop_idx, tfrecord_file):
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for idx in range(start_idx, stop_idx):
        tfeat, ttarg = get_binary_data(reader, idx)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'features': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[tfeat])
                    ),
                    'targets': tf.train.Feature(
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
            'features': tf.FixedLenFeature([], tf.string),
            'targets': tf.FixedLenFeature([], tf.string),
        },
        name='data'
    )
    tfeat = tf.decode_raw(tfrecord_features['features'], tf.uint8)
    # note, 'NCHW' is only supported on GPUs, so use 'NHWC'...
    tfeat = tf.reshape(tfeat, [-1, 28, 28, 1])
    ttarg = tf.decode_raw(tfrecord_features['targets'], tf.uint8)
    ttarg = tf.one_hot(indices=ttarg, depth=10, on_value=1, off_value=0)
    return tfeat, ttarg


def batch_generator(filenames, batch_size=32, num_epochs=1):
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
        except tf.errors.OutOfRangeError:
            print('out of examples')
        except Exception as e:
            print(e)
        finally:
            coord.request_stop()
            coord.join(threads)


def write_all(train_file, valid_file, test_file):
    m = mnist_hdf5_reader('mnist.hdf5')
    m.open()
    if not os.path.isfile(train_file + '.gz'):
        write_tfrecord(m, 0, 50000, train_file)
        gz_compress(train_file)
    if not os.path.isfile(valid_file + '.gz'):
        write_tfrecord(m, 50000, 60000, valid_file)
        gz_compress(valid_file)
    if not os.path.isfile(test_file + '.gz'):
        write_tfrecord(m, 60000, 70000, test_file)
        gz_compress(test_file)
    m.close()


def read_all(train_file, valid_file, test_file):
    test_read_tfrecord(train_file + '.gz')
    test_read_tfrecord(valid_file + '.gz')
    test_read_tfrecord(test_file + '.gz')


if __name__ == '__main__':
    train_file = 'mnist_train.tfrecord'
    valid_file = 'mnist_valid.tfrecord'
    test_file = 'mnist_test.tfrecord'
    write_all(train_file, valid_file, test_file)
    read_all(train_file, valid_file, test_file)
