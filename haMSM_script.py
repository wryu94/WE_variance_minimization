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
iter_last = 2000

# Define bin boundaries (in pcoord, 1D bin for now) in WESTPA style and number of bins
# Make sure to have the infinity as np.inf
bin_boundaries = [0.0,1.00] + \
		 [1.10+0.1*i for i in range(35)] + \
		 [4.60+0.2*i for i in range(10)] + \
		 [6.60+0.6*i for i in range(6)] + \
		 [np.inf]
n_bins = len(bin_boundaries)-1

# Number of MSM clusters to have in each stratum/WE bin when doing stratified clustering 
# Details of stratified clustering shown in https://mr-toolkit.readthedocs.io/en/latest/_examples/stratified_clustering.html
# First and last bin (assumed to be either target or basis bin) are assigned single clusters
# Same indexing as `bin_boundaries`
clusters_per_bin_fill_value = 5
clusters_per_bin = np.full(n_bins, 
                           clusters_per_bin_fill_value,
                           dtype=int)
clusters_per_bin[0] = 1
clusters_per_bin[-1] = 1

# WE simulation data list
h5_file_paths = ['/home/groups/ZuckermanLab/ryuwo/NTL9_WE_benchmark/ntl9_WE_low_friction_gamma_5_inv_ps/rep1/west.h5']

# Dimensionality reduction method
# Option of 'vamp', 'tica', and 'none'
dimreduce_method = 'vamp'

# Boundaries of the basis/target, in progress coordinate space
pcoord_bounds = {
    'basis': [[9.6, 100.0]], 
    'target': [[-0.01, 1.0]]
}

# Model name
model_name = 'NTL9_folding'

# Reference structure
# Actually isn't used here, but make sure you have a correct reference structure in the haMSM_functions.py file
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
###                          > intermediate_dimreduce(coord) 
###                          > dimreduce_model.transform(intermediate_dimreduce(coord))
###                          > pcoord 
###    (Low dimensionality)
####################################################################################################
####################################################################################################

# Train the dimensionality reduction model
# You can use train_dimreduce_continue to further train a model that you already have 
print('Training dimreduce model')
dimreduce_model = train_dimreduce_start(
    dimreduce_method, 
    h5_file_paths,
    iter_first,
    iter_last
)
with open('dimreduce_model.pkl', 'wb') as file:
    pickle.dump(dimreduce_model, file)

# Get concatenated WE data
print('Concatenating WE data')
WE_data_concat = concatenate_WE_data(
    h5_file_paths, 
    iter_first,
    iter_last,
    dimreduce_model
)
print(WE_data_concat)
with open('WE_data_concat.pkl', 'wb') as file:
    pickle.dump(WE_data_concat, file)

# Uncomment the block below to read in data
"""
# Load dimreduce mode
with open("dimreduce_model.pkl", 'rb') as file:
    dimreduce_model = pickle.load(file)

# Load concatenated data
with open("WE_data_concat.pkl", 'rb') as file:
    WE_data_concat = pickle.load(file)
"""

# Get kmeans and the cluster centers 
# `stratified_clustering_return_Kmeans` function places only single cluster for 
# first and last bin (assumed to be either source or sink and sink or source)
print('Getting stratified clustering objects and cluster centers')
kmeans_models, cluster_centers = \
stratified_clustering_return_Kmeans(bin_boundaries,
                                    clusters_per_bin,
                                    WE_data_concat)

# Calculate the flux matrix 
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
np.savetxt('flux_mtx.txt',hamsm_flux_matrices[0])
np.savetxt('cluster_centers.txt',cluster_centers)
with open('kmeans_models.pkl', 'wb') as file:
    pickle.dump(kmeans_models, file)

# Note: flux matrix is the fundamental(?) quantity that you calculate from the WE simulation.
# Since flux matrix is just an averaged value of all the iterations, if you want to get
# a flux matrix from more than one WE simulation, you simply average the two (or how many replicates you're using)
# Meaning that the 'best' haMSM will be the average of all the h5 files
# But here, since I'm just building a pipeline, I'll do an analysis of each replicates / you can do 
# further analysis to get results from more than one replicate

# Block below calculates the transition matrix, steady state distribution, and other quantities
# You can change the post-processing of the flux matrix based on what you want to do
"""
# Getting target and source cluster center
target_cluster_index = np.where((cluster_centers>pcoord_bounds['target'][0][0]) & (cluster_centers<pcoord_bounds['target'][0][1]))[0][0]
basis_cluster_index = np.where((cluster_centers>pcoord_bounds['basis'][0][0]) & (cluster_centers<pcoord_bounds['basis'][0][1]))[0][0]

# Calculate transition matrix 
transition_matrices = []
for flux_matrix in hamsm_flux_matrices:
    transition_matrix = get_Tmatrix(flux_matrix, target_cluster_index, basis_cluster_index)
    transition_matrices += [transition_matrix]

# Calculate functions
# Again, now we're saving these function for each replicate, so this is 2d array 
pi = []
h = []
v = []
for T in transition_matrices:
    msm = MarkovStateModel(T)
    pi_rep = msm.stationary_distribution
    h_rep, v_rep = solve_discrepancy(T, pi_rep, [target_cluster_index])

    pi += [pi_rep]
    h  += [h_rep]
    v  += [v_rep]
"""

# Done!
print('Done!')
