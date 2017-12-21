from __future__ import print_function
import h5py
import os


def slices_maker(n, slice_size=100000):
    """
    make "slices" of size `slice_size` from a file of `n` events
    (so, [0, slice_size), [slice_size, 2 * slice_size), etc.)
    """
    if n < slice_size:
        return [slice(0, n)]

    remainder = n % slice_size
    n = n - remainder
    nblocks = n // slice_size
    counter = 0
    slices = []
    for i in range(nblocks):
        end = counter + slice_size
        slices.append(slice(counter, end))
        counter += slice_size

    if remainder != 0:
        slices.append(slice(counter, counter + remainder))

    return slices


class HDF5Reader:
    """
    user should call `open()` and `close()` to start/finish.
    """
    def __init__(self, hdf5_file, batch_size=50):
        self.file = hdf5_file
        self._f = None
        self._batch_size = batch_size

    def open(self):
        self._f = h5py.File(self.file, 'r')
        self._cursor = 0
        self._nevents = self._f['targets'].shape[0]

    def close(self):
        try:
            self._f.close()
            self._cursor = 0
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

    # def __iter__(self):
    #     """ this version of the function uses the `slices_maker` function """
    #     slcs = slices_maker(self._nevents, self._batch_size)
    #     for slc in slcs:
    #         targets = self._f['targets'][slc]
    #         features = self._f['features'][slc]
    #         yield targets, features

    def __iter__(self):
        """ this version doesn't use `slices_maker()` """
        while True:
            if self._cursor >= self._nevents:
                break
            slc = slice(self._cursor, self._cursor + self._batch_size)
            targets = self._f['targets'][slc]
            features = self._f['features'][slc]
            yield targets, features
            self._cursor += self._batch_size


if __name__ == '__main__':
    h5f = os.environ['HOME'] + '/Dropbox/Data/RandomData/hdf5/mnist_test.hdf5'
    reader = HDF5Reader(h5f)
    reader.open()
    try:
        it = iter(reader)
        while True:
            t, f = next(it)
            print(t.shape, f.shape)
    except Exception as e:
        print(e)
    finally:
        reader.close()
