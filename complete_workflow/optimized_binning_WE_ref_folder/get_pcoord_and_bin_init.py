import numpy as np
import pickle
from haMSM_functions import get_Tmatrix, solve_discrepancy, haMSM_result_aggregated, featurization, stratified_clustering
from deeptime.markov.msm import MarkovStateModel
from joblib import load
import mdtraj as md

# Read in RMSD value from cpptraj output file `rmsd.temp`
pcoord_1 = np.loadtxt('rmsd.temp')[1]

# Parameter file, this will be constant for the NTL9 system
WE_ref_file     = "/home/groups/ZuckermanLab/ryuwo/NTL9_WE_benchmark/ntl9_WE_optimized/low_friction_gamma_5_inv_ps_tau_100_ps_pair_dist_featurization/rep_optimization_test/ref_files/"
param_file  = WE_ref_file + 'ntl9.prmtop'
# Restart file, eventually turn this into input 
restart_file = WE_ref_file + 'ntl9.rst7'
#restart_file  = 'parent.rst7'

# Bin boundaries from unoptimized WE 
bin_boundaries = [0.0,1.00] + [1.10+0.1*i for i in range(35)] + [4.60+0.2*i for i in range(10)] + [6.60+0.6*i for i in range(6)] + [float('inf')]
n_bins = len(bin_boundaries)-1

# Dimreduce and kmeans model path
# Also read in cluster vs bin index results here
haMSM_path = "/home/groups/ZuckermanLab/ryuwo/NTL9_WE_benchmark/ntl9_haMSM_models/low_friction/pair_featurization/"
dimreduce_path = haMSM_path + '1_train_dimreduce/dimreduce_model.joblib'
kmeans_path = haMSM_path + '3_flux_matrix_calc_save_all_flux_mtx/kmeans_models.joblib'
cluster_vs_binning_path = haMSM_path+"4_post_processing/cluster_vs_binning.npy"
# Load models 
dimreduce_model = load(dimreduce_path)
kmeans_models   = load(kmeans_path)
# Get binning (precalculated)
binning         = np.load(cluster_vs_binning_path)

# Get system coordinate u_pos
def mdtraj_load_pos(
    restart_path,
    topology_path,
    selection='protein'
):
    mdtraj_u = md.load(restart_path, top=topology_path)
    selected_atoms = mdtraj_u.topology.select(selection)
    mdtraj_u_pos = mdtraj_u.xyz[0, selected_atoms, :] * 10.0
    return mdtraj_u_pos

# Get system coordinate u_pos
u_pos_1 = mdtraj_load_pos(restart_file, param_file)

# Predict cluster assignment 
dimreduce_1 = dimreduce_model.transform(featurization(u_pos_1))#[0])
X_1 = np.concatenate(([pcoord_1], dimreduce_1))
cluster_assignment_1 = stratified_clustering(bin_boundaries, kmeans_models, X_1)

# Get bin assignment
bin_assignment_1 = binning[cluster_assignment_1]

# Save as pcoord_and_bin
np.savetxt('pcoord_and_bin.dat', np.column_stack((pcoord_1, bin_assignment_1)), fmt='%.6f %d')
