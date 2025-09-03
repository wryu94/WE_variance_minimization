#!/bin/bash

if [ -n "$SEG_DEBUG" ] ; then
    set -x
    env | sort
fi

cd $WEST_SIM_ROOT
echo 'I am here'

pwd

#cp ntl9.rst7 parent.rst7
echo "west data ref: $WEST_STRUCT_DATA_REF" > wsdata.txt
cp $WEST_STRUCT_DATA_REF ./parent.rst7
function cleanup() {
    rm -f parent.rst7
   #rm -f phi.dat psi.dat
}

trap cleanup EXIT

# Get progress coordinate
cpptraj ref_files/ntl9.prmtop < ref_files/ptraj_init.in || exit 1
ln -s ref_files/get_pcoord_and_bin_init.py ./
python get_pcoord_and_bin_init.py
cat pcoord_and_bin.dat > $WEST_PCOORD_RETURN

if [ -n "$SEG_DEBUG" ] ; then
    head -v $WEST_PCOORD_RETURN
fi

