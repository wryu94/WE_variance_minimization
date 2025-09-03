import numpy as np
from deeptime.markov.msm import MarkovStateModel
import pickle
import warnings
# To not see the warning when you calculate CA distances using MDAnalaysis
warnings.filterwarnings("ignore")

from haMSM_functions import concatenate_WE_data, \
                    stratified_clustering_return_Kmeans, \
                    get_fluxMatrix, \
                    get_Tmatrix, \
                    train_dimreduce_start, \
                    train_dimreduce_continue, \
                    solve_discrepancy

####################################################################################################
####################################################################################################
### haMSM parameters
####################################################################################################
####################################################################################################

# First and last iterations to use
# This range of iterations should be present in your WE h5 files
iter_first = 1
iter_last = 450

# Define bin boundaries (in pcoord, 1D bin for now) and number of bins
# Make sure to have the infinity as np.inf
# In WESTPA style 
bin_boundaries = [0.0,1.00] + \
		 [1.10+0.1*i for i in range(35)] + \
		 [4.60+0.2*i for i in range(10)] + \
		 [6.60+0.6*i for i in range(6)] + \
		 [np.inf]
n_bins = len(bin_boundaries)-1

# Number of MSM clusters to have in each stratum/WE bin 
# First and last bin (assumed to be either target or basis bin) 
# are assigned single cluster 
# Same indexing as `bin_boundaries`
clusters_per_bin_fill_value = 5
clusters_per_bin = np.full(n_bins, 
                           clusters_per_bin_fill_value,
                           dtype=int)
clusters_per_bin[0] = 1
clusters_per_bin[-1] = 1

# WE simulation data list
we_path = "/home/groups/ZuckermanLab/ryuwo/NTL9_WE_benchmark/ntl9_WE_low_friction_gamma_5_inv_ps_tau_100ps/"
h5_file_paths = \
[we_path+'rep1/west.h5',
 we_path+'rep2/west.h5',
 we_path+'rep3/west.h5',
 we_path+'rep4/west.h5',
 we_path+'rep5/west.h5']

# Dimensionality reduction method
# Option of 'vamp', 'tica', and 'none'
dimreduce_method = 'vamp'

# Boundaries of the basis/target, in progress coordinate space
pcoord_bounds = {
    'basis': [[9.6, 100.0]], 
    'target': [[-0.01, 1.0]]
}

# Model name
model_name = 'NTL9 folding'

# Reference structure
# Actually isn't used here, but make sure you have a correct reference structure in the function.py file
ref_file = 'reference.pdb'

####################################################################################################
####################################################################################################
### haMSM pipeline
### 
### 1. Train the dimensionality reduction model on the transition data from the
###    original h5 file. 
### 2. Concatenate data (every structure observed from WE) into a dict that contains
###    pcoord: progress coordinate
###    dimreduce: dimreduced on the alpha carbon pairwise distance which is calculated from full structures
###    weight: weight of that structure
### 3. Using the concatenated data, train stratified clustering model 
### 4. Using the dimreduce model and the stratified clustering model, construct the flux matrix 
###    from h5 WE data.
### 5. From the raw flux matrix, calculate transition matrix, steady state distribution, 
###    discrepancy, variance, and other quantities of interest. 
###
### Notes:
### -  The auxdata/coord in the h5 file is the full coordinate of the system. For the NTL9 case,
###    since it's an implicit solvent system, it's the NTL9 protein coordinate. But when you read in 
###    this data, you usually use the pairwise CA (alpha carbon) distance, not the full atomic coordinate. 
###    This pairwise distance is different from pcoord (progress coordinate), which is set to be 
###    the RMSD wrt the folded structure for the NTL9 system. The dimreduce (VAMP, TICA, etc) models are
###    trained on these pairwise distances. So the 4 coordinates that we use in haMSM pipeline are:
###
###    (High dimensionality) coord 
###                          > pairwise_dist(coord) 
###                          > dimreduce_model.transform(pairwise_dist(coord))
###                          > pcoord 
###    (Low dimensionality)
####################################################################################################
####################################################################################################

# Load dimreduce model
dimreduce_path = "/home/groups/ZuckermanLab/ryuwo/NTL9_WE_benchmark/ntl9_haMSM_pair_dist_featurization/low_friction/1_train_dimreduce/dimreduce_model.pkl"
with open(dimreduce_path, 'rb') as file:
    dimreduce_model = pickle.load(file)

# Load concatenated data
WE_data_path = "/home/groups/ZuckermanLab/ryuwo/NTL9_WE_benchmark/ntl9_haMSM_pair_dist_featurization/low_friction/2_train_stratified_clustering/WE_data_concat.pkl"
with open(WE_data_path, 'rb') as file:
    WE_data_concat = pickle.load(file)

# Get kmeans and the cluster centers 
# `stratified_clustering_return_Kmeans` function places only single cluster for 
# first and last bin (assumed to be either source or sink and sink or source)
print('Getting stratified clustering objects and cluster centers')
kmeans_models, cluster_centers = \
stratified_clustering_return_Kmeans(bin_boundaries,
                                    clusters_per_bin,
                                    WE_data_concat)

# Flux matrix calculation
# Note that `hamsm_flux_matrices` is a list of matrices (each from replicate)
print('Calculating flux matrix')
hamsm_flux_matrices = \
get_fluxMatrix(
    h5_file_paths,
    cluster_centers,
    kmeans_models,
    iter_first,
    iter_last,
    bin_boundaries,
    dimreduce_model
)

# Save result
with open('flux_matrices.pkl', 'wb') as file:
    pickle.dump(hamsm_flux_matrices, file)
np.savetxt('cluster_centers.txt',cluster_centers)
with open('kmeans_models.pkl', 'wb') as file:
    pickle.dump(kmeans_models, file)

