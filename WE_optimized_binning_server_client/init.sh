#!/bin/bash
#source env.sh

rm -rf seg_logs traj_segs
rm -f west.h5
mkdir seg_logs traj_segs 

BSTATE_ARGS="--bstate-file bstates.txt"
TSTATE_ARGS="--tstate folded,0.99,52"  # second coord is (ad hoc) optimized bin and 52 is just the bin index of the target bin 
w_init $BSTATE_ARGS $TSTATE_ARGS --segs-per-state 4 "$@" | tee init.log
