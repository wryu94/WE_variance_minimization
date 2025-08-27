#!/bin/bash
if [ -n "$SEG_DEBUG" ] ; then
    set -x
    env | sort
fi
cd $WEST_SIM_ROOT

mkdir -pv $WEST_CURRENT_SEG_DATA_REF || exit 1
cd $WEST_CURRENT_SEG_DATA_REF || exit 1

# Set up the run
ln -sv $WEST_SIM_ROOT/ref_files/{reference.pdb,ntl9.prmtop,ptraj.in,ptraj_rg.in,ptraj_surfarea.in} .

case $WEST_CURRENT_SEG_INITPOINT_TYPE in
    SEG_INITPOINT_CONTINUES)
        # A continuation from a prior segment
        sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/ref_files/md-continue.in > md.in
        ln -sv $WEST_PARENT_DATA_REF/seg.rst7 ./parent.rst7
    ;;

    SEG_INITPOINT_NEWTRAJ)
        # Initiation of a new trajectory; $WEST_PARENT_DATA_REF contains the reference to the
        # appropriate basis state or generated initial state
        sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/ref_files/md-genvel.in > md.in
        if [ $WEST_RUN_STATUS="Init" ] ; then
            ln -sv $WEST_PARENT_DATA_REF parent.rst7
            echo "linking $WEST_PARENT_DATA_REF"
        else
            ln -sv $WEST_SIM_ROOT/ref_files/ntl9.rst7 parent.rst7
        fi
    ;;

    *)
        echo "unknown init point type $WEST_CURRENT_SEG_INITPOINT_TYPE"
        exit 2
    ;;
esac

# Propagate segment
     pmemd.cuda -O -i md.in \
               -p ntl9.prmtop     \
               -c parent.rst7    \
               -r seg.rst7       \
               -o seg.out        \
               || exit 1


# Get progress coordinate
cpptraj ntl9.prmtop < ptraj.in || exit 1
ln -s $WEST_SIM_ROOT/ref_files/get_pcoord_and_bin_run.py ./
python get_pcoord_and_bin_run.py 
cat pcoord_and_bin.dat > $WEST_PCOORD_RETURN


