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
# Hbond appears unused
#ln -sv $WEST_SIM_ROOT/ref_files/{reference.pdb,ntl9.prmtop,ptraj.in,hbond_ptraj.in,ptraj_rg.in,ptraj_surfarea.in} .

# Set up the run
#ln -sv $WEST_SIM_ROOT/{reference.pdb,proteinG.prmtop,ptraj.in,hbond_ptraj.in,ptraj_rg.in,ptraj_surfarea.in} .

case $WEST_CURRENT_SEG_INITPOINT_TYPE in
    SEG_INITPOINT_CONTINUES)
        # A continuation from a prior segment
        sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/ref_files/md-continue.in > md.in
        ln -sv $WEST_PARENT_DATA_REF/seg.rst7 ./parent.rst7
#        ln -sv $WEST_SIM_ROOT/md-continue.in md.in
    ;;

    SEG_INITPOINT_NEWTRAJ)
        # Initiation of a new trajectory; $WEST_PARENT_DATA_REF contains the reference to the
        # appropriate basis state or generated initial state
        sed "s/RAND/$WEST_RAND16/g" $WEST_SIM_ROOT/ref_files/md-genvel.in > md.in
#        ln -sv $WEST_SIM_ROOT/md-genvel.in md.in
        if [ $WEST_RUN_STATUS="Init" ] ; then
#            ln -sv $WEST_SIM_ROOT/ntl9.rst7 parent.rst7
            ln -sv $WEST_PARENT_DATA_REF parent.rst7
            echo "linking $WEST_PARENT_DATA_REF"
        else
            ln -sv $WEST_SIM_ROOT/ref_files/ntl9.rst7 parent.rst7
#            ln -sv $WEST_PARENT_DATA_REF parent.rst7
#            echo "linking $WEST_PARENT_DATA_REF"
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
cpptraj ntl9.prmtop <ptraj.in || exit 1
gawk '{print $2}' rmsd.temp | tail -2 > pcoord.dat || exit 1
cat pcoord.dat > $WEST_PCOORD_RETURN

#cpptraj ntl9.prmtop < hbond_ptraj.in || exit 1
#gawk '{print $2}' hbond.dat | tail -2 > hbond2.dat || exit 1
#cat hbond2.dat > $WEST_HBOND_RETURN
#
#cpptraj ntl9.prmtop < ptraj_rg.in || exit 1
#gawk '{print $2}' rog.dat | tail -2 > rog2.dat || exit 1
#cat rog2.dat > $WEST_RG_RETURN
#
#cpptraj ntl9.prmtop < ptraj_surfarea.in || exit 1
#gawk '{print $2}' sa.dat | tail -2 > sa2.dat || exit 1
#cat sa2.dat > $WEST_SURFAREA_RETURN
#
#rm mdinfo rmsd.temp hbond* rog* sa.dat sa2.dat 

