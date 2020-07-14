#!/bin/bash
# Use the hdf5_manipulator package:
#  https://github.com/gnperdue/hdf5_manipulator

DATADIR="/Users/perdue/Dropbox/Data/RandomData/StarGalaxy"
FILELIST="
stargalaxy_real_pt_test
stargalaxy_real_pt_train
stargalaxy_real_pt_valid
"

for file in $FILELIST
do
  python hdf5_manipulator/uint8float_rescaler.py --input ${DATADIR}/${file}.hdf5 \
    --output ${file}_flt.hdf5 \
    --imgnames imageset
done

mv stargalaxy_real_pt_test_flt.hdf5 stargalaxy_real_ptflt_test.hdf5
mv stargalaxy_real_pt_train_flt.hdf5 stargalaxy_real_ptflt_train.hdf5
mv stargalaxy_real_pt_valid_flt.hdf5 stargalaxy_real_ptflt_valid.hdf5

