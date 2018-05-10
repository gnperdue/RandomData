import tensorflow as tf


def make_batch_generator_fn(batch_size=10, dset_size=100):
    feats, targs = range(dset_size), range(1, dset_size + 1)

    def batch_generator_fn():
        start_idx, stop_idx = 0, batch_size
        while True:
            if stop_idx > dset_size:
                return
            yield feats[start_idx: stop_idx], targs[start_idx: stop_idx]
            start_idx, stop_idx = start_idx + batch_size, stop_idx + batch_size

    return batch_generator_fn


def test(batch_size=10):
    dgen = make_batch_generator_fn(batch_size)
    features_shape, targets_shape = [None], [None]
    ds = tf.data.Dataset.from_generator(
        dgen, (tf.int32, tf.int32),
        (tf.TensorShape(features_shape), tf.TensorShape(targets_shape))
    )
    feats, targs = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        counter = 0
        try:
            while True:
                f, t = sess.run([feats, targs])
                print(f, t)
                counter += 1
                if counter > 15:
                    break
        except tf.errors.OutOfRangeError:
            print('end of dataset at counter = {}'.format(counter))


if __name__ == '__main__':
    test()
