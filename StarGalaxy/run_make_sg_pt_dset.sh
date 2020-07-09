#!/bin/bash
# Use the hdf5_manipulator package:
#  https://github.com/gnperdue/hdf5_manipulator

DATADIR="/Users/perdue/Dropbox/Quantum_Computing/hep-qml/data"
FILELIST="
stargalaxy_real
"

for file in $FILELIST
do
  python hdf5_manipulator/tftnsr2pttnsr.py --input ${DATADIR}/${file}.h5 \
    --output ${file}_pt.hdf5 \
    --tensor imageset
  python hdf5_manipulator/split_big_special.py --input "${file}_pt.hdf5" --size 0
done
