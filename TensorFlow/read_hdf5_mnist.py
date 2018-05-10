from __future__ import print_function
from __future__ import absolute_import

import h5py


class HDF5Reader:
    """
    user should call `openf()` and `closef()` to start/finish.
    """
    def __init__(self, hdf5_file, batch_size=50):
        self._file = hdf5_file
        self._f = None
        self._batch_size = batch_size

    def openf(self):
        self._f = h5py.File(self._file, 'r')
        self._nevents = self._f['targets'].shape[0]
        return self._nevents

    def closef(self):
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
