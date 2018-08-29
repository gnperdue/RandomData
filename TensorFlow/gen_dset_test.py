import tensorflow as tf


def make_batch_generator_fn(dset_size=100):
    feats, targs = range(dset_size), range(1, dset_size + 1)

    def batch_generator_fn():
        idx = 0
        while True:
            if idx >= dset_size:
                return
            yield feats[idx], targs[idx]
            idx += 1

    return batch_generator_fn


def test():
    dgen = make_batch_generator_fn()
    features_shape, targets_shape = (), ()
    ds = tf.data.Dataset.from_generator(
        dgen, (tf.int32, tf.int32),
        (tf.TensorShape(features_shape), tf.TensorShape(targets_shape))
    )
    ds = ds.shuffle(20)
    ds = ds.batch(10)
    feats, targs = ds.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        counter = 0
        try:
            while True:
                f, t = sess.run([feats, targs])
                print(f.shape, t.shape)
                print(f.reshape(-1,), t.reshape(-1,))
                counter += 1
                if counter > 15:
                    break
        except tf.errors.OutOfRangeError:
            print('end of dataset at counter = {}'.format(counter))


if __name__ == '__main__':
    test()
