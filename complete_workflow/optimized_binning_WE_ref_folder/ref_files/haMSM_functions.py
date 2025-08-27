import h5py
import numpy as np
import pickle
import deeptime
from sklearn.cluster import KMeans
from deeptime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM
import mdtraj as md
from scipy.sparse import coo_matrix
from MDAnalysis.analysis import rms
from tqdm import tqdm
from deeptime.decomposition import TICA, VAMP

# Reference structure to be used in intermediate_dimreduce
ref_file = 'reference.pdb'

def featurization(coords):
# Pairwise distance for CA using mdtraj

    # Load reference structure in mdtraj
    reference = md.load(ref_file)

    # Get alpha carbon indices and pairs
    CA_indices = reference.top.select_atom_indices('alpha')
    CA_pairs = [(i, j) for idx_i, i in enumerate(CA_indices)
                     for j in CA_indices[idx_i+1:]]

    # Make mdtraj object from the coord
    traj = md.Trajectory(coords / 10.0, reference.top)

    # Calculate pairwise distance (alpha carbon) 
    # Compute pairwise distances for CA atoms
    dist_arr = md.compute_distances(traj, CA_pairs)[0] * 10.0  # shape: (n_pairs,)

    # Return result
    return dist_arr

def processCoordinates(coords):
    """
    Calculates the Root Mean Square Deviation (RMSD) of the given coordinates with respect to a reference structure.

    Comment: this function isn't really used in the haMSM script, but I've added here just in case. 

    Args:
    - coords: 2D list that contains your system's coordinate. Must have same dimensionality as what's in ref_file. 
    - ref_file: Reference pdb file that has the reference structure of your system.

    Returns:
    - rmsd: NumPy array containing RMSD values for each frame in the coordinates.
    """
    # Load reference structure in mdtraj
    reference = md.load(ref_file)

    # Get alpha carbon indices
    alpha_indices = reference.top.select_atom_indices('alpha')
    
    # Make mdtraj object from the coord
    check = md.Trajectory(coords, reference.top)
    
    # Calculate rmsd (alpha carbon) 
    # 10 for unit conversion (nm to Angstrom)
    rmsd = md.rmsd(reference, check, atom_indices=alpha_indices) * 10
    
    # Return result
    return rmsd

class IdentityTransformModel:
    # Dummy class in case you're not using any dimreduce ('none' option) 
    def transform(coord):
        return coord

def train_dimreduce_start(
    dimreduce_method, # 'vamp', 'tica', or 'none' for now
    h5_file_paths,
    iter_first,
    iter_last
):
    """
    Trains a dimensionality reduction model using VAMP, TICA, or none on iteration data from HDF5 files.

    Comment: This procedure doesn't really fully use the WE data because neither TICA or VAMP can use weight in their training. 
    But seems like this is the best we can do now. It was similarly implemented in msm_we

    Args:
    - dimreduce_method: String, the dimensionality reduction method to use ('vamp', 'tica', or 'none').
    - h5_file_paths: List of paths to HDF5 files containing iteration data.
    - iter_first: Integer, the starting iteration for training.
    - iter_last: Integer, the ending iteration for training.

    Returns:
    - model: Trained dimensionality reduction model based on the specified method.
    """
    # Define the model 
    if dimreduce_method=='vamp':
        model = VAMP(lagtime=1, var_cutoff=0.95, scaling="kinetic_map")
    elif dimreduce_method=='tica':
        model = TICA(lagtime=1, var_cutoff=0.95, scaling="kinetic_map")
    elif dimreduce_method=='none': # This would just use straight up CA distances 
        model = IdentityTransformModel
        return model

    # Loop through the provided h5 files
    for h5_file in h5_file_paths:
        print(f'Processing {h5_file}')

        with h5py.File(h5_file, "r") as data_file:
            for iteration in tqdm(range(iter_first,iter_last+1)):
                # Coordinates from and to (start and end of traj)
                coord_from = data_file['iterations'][f'iter_{iteration:08d}']['auxdata']['coord'][:][:,0,:,:]
                coord_to = data_file['iterations'][f'iter_{iteration:08d}']['auxdata']['coord'][:][:,1,:,:]

                # Calculate CA distance for each structure
                pair_dist_from = np.array([intermediate_dimreduce(coord)[0] for coord in coord_from])
                pair_dist_to = np.array([intermediate_dimreduce(coord)[0] for coord in coord_to])

                # Do partial training of the model
                model.partial_fit((pair_dist_from,
                                   pair_dist_to))

    # Return model
    return model

def train_dimreduce_continue(
    model, # Previous dimreduce model that you trained, assuming it's not 'none'
    h5_file_paths,
    iter_first,
    iter_last
):
    """
    Continues training a dimensionality reduction model using iteration data from HDF5 files.

    Args:
    - model: Previously trained dimensionality reduction model (VAMP or TICA).
    - h5_file_paths: List of paths to HDF5 files containing iteration data.
    - iter_first: Integer, the starting iteration for training.
    - iter_last: Integer, the ending iteration for training.

    Returns:
    - model: Updated dimensionality reduction model after continuation of training.
    """
    # Loop through the provided h5 files
    for h5_file in h5_file_paths:
        print(f'Processing {h5_file}')

        with h5py.File(h5_file, "r") as data_file:
            for iteration in tqdm(range(iter_first,iter_last+1)):
                # Coordinates from and to (start and end of traj)
                coord_from = data_file['iterations'][f'iter_{iteration:08d}']['auxdata']['coord'][:][:,0,:,:]
                coord_to = data_file['iterations'][f'iter_{iteration:08d}']['auxdata']['coord'][:][:,1,:,:]

                # Calculate CA distance for each structure
                pair_dist_from = np.array([intermediate_dimreduce(coord)[0] for coord in coord_from])
                pair_dist_to = np.array([intermediate_dimreduce(coord)[0] for coord in coord_to])

                # Do partial training of the model
                model.partial_fit((pair_dist_from,
                                   pair_dist_to))

    # Return model
    return model

def concatenate_WE_data(
    h5_file_paths,
    iter_first,
    iter_last, 
    dimreduce_model
):
    """
    Directly accesses the west.h5 files and collects the WE data (pcoord and dimreduce
    calculated from coord (pairwise CA distance and dimreduce on that using the model
    that was trained)).

    Args:
    - h5_file_paths: List of paths to west.h5 files.
    - iter_first: Integer, the starting iteration for data collection.
    - iter_last: Integer, the ending iteration for data collection.
    - dimreduce_model: Trained dimensionality reduction model for transforming coordinates.

    Returns:
    - WE_data_concat: Dictionary containing concatenated WE data with keys as dummy int values.
        Each entry in the dictionary contains:
            - 'pcoord': Progress coordinates for the trajectory.
            - 'dimreduce': Dimensionally reduced coordinates based on the trained model.
            - 'weight': Weight of the trajectory.
    """
    # Define the concatenated dataset to return 
    WE_data_concat = {}

    # Counting the snapshots to save in the concatenated data 
    counter = 0 

    for h5_file in h5_file_paths:
        print(f'Processing {h5_file}')

        with h5py.File(h5_file, "r") as data_file:
            for iteration in tqdm(range(iter_first,iter_last+1)):
                # Save the pcoord and coord from this iteration 
                # Note that we're saving the pcoord and coord AFTER the iteration is run
                # hence [:,-1,:] for pcoord and [:,-1,:,:] for coord as per 
                # https://github.com/jdrusso/msm_we/blob/7e346567d246a718d5f8eeff08aa2a5cc3d04b68/msm_we/_hamsm/_data.py#L616
                pcoord = data_file['iterations'][f'iter_{iteration:08d}']['pcoord'][:][:,-1,:]
                coord = data_file['iterations'][f'iter_{iteration:08d}']['auxdata']['coord'][:][:,-1,:,:]

                # Weights of these trajectories
                full_weight_data = data_file['iterations'][f'iter_{iteration:08d}']['seg_index'][:]
                weight = [item[0] for item in full_weight_data]

                # Number of trajectories in this iteration
                n_traj = np.shape(pcoord)[0]

                # Loop through pcoord and coord and add the info in the WE_data_concat
                # Note that key of WE_data_concat is a dummy int value 
                # Prob could have WE_data_concat in a list instead of dict, but kinda lazy 
                for i_traj in range(n_traj):
                    # Save pcoord, coord, weight in `WE_data_concat`
                    WE_data_concat[counter] = {
                        'pcoord': pcoord[i_traj][0],
                        'dimreduce': dimreduce_model.transform(intermediate_dimreduce(coord[i_traj])[0]),
                        'weight': weight[i_traj]
                    }

                    # Update counter 
                    counter += 1 

    # Return dict
    return WE_data_concat

def stratified_clustering_return_Kmeans(
    bin_boundaries,
    clusters_per_bin,
    WE_data_concat
):
    """
    Perform stratified clustering based on progress coordinate and return KMeans models.

    Args:
    - bin_boundaries (list): A list of boundaries for stratification based on progress coordinate.
    - clusters_per_bin (list): The number of clusters to create per bin.
    - WE_data_concat (dict): A dictionary containing concatenated weighted ensemble data.

    Returns:
    - stratified_kmeans (list): A list of KMeans clustering models for each stratum, with the first and last strata having a single cluster.
    - cluster_centers (np.ndarray): An array of cluster centers.

    This function performs stratified clustering based on progress coordinate and returns a list of KMeans models, one for each stratum.
    The number of clusters is specified for each stratum, except for the first and last strata where only one cluster is created.
    It also returns an array of cluster centers.
    """
    # Number of bins
    n_bin = len(bin_boundaries) - 1

    # This list will contain KMeans objects
    stratified_kmeans = []

    # This will contain cluster center RMSDs
    cluster_centers = []

    for i in range(n_bin):
        print(f'processing bin {i}')

        # Make a dictionary of data points that are in this bin 
        data_point_in_this_bin = {}

        # Counter for key to be used in data_point_in_this_bin
        counter = 0 

        for key, item in WE_data_concat.items():
            # Based on progress coordinate/stratified clustering
            if bin_boundaries[i] < item['pcoord'] and item['pcoord'] < bin_boundaries[i + 1]:
                data_point_in_this_bin[counter] = item
                counter += 1

        # Now we want to get X (pcoord and dimreduce) and weight
        # This probably won't be necessary for AA WE runs, but may need it for synD
        # We check whether same pcoord and dimreduce was included in X
        # If so, we only update the weight
        # If not, add to X
        X = []
        weight = []
        for key, item in data_point_in_this_bin.items():
            array_i = np.concatenate(([item['pcoord']], item['dimreduce']))
            weight_i = item['weight']

            if X and np.any(np.all(X == array_i, axis=1)):
                index = np.where(np.all(X==array_i, axis=1))[0][0]
                weight[index] += weight_i
            else:
                X += [array_i]
                weight += [weight_i]

        print(f'There are {len(weight)} data points in this bin')
        
        # Define KMeans object and train on the data in this bin 
        kmeans = KMeans(n_clusters=clusters_per_bin[i], n_init='auto', random_state=53982).fit(X=X, y=None, sample_weight=weight)

        # Save the model in the list
        stratified_kmeans += [kmeans]

        # Save cluster centers
        cluster_centers += [kmeans.cluster_centers_[:, 0]]

    # Concatenate into a single list 
    cluster_centers = np.concatenate(cluster_centers)

    return stratified_kmeans, cluster_centers

def get_fluxMatrix(
    h5_file_paths,
    cluster_centers,
    kmeans_models,
    iter_first,
    iter_last,
    bin_boundaries,
    dimreduce_model
):    
    """
    Calculate flux matrices from weighted ensemble data for each replicate.

    Args:
    - h5_file_paths (list): List of paths to west.h5 files.
    - cluster_centers (np.ndarray): Array of cluster centers obtained from stratified clustering.
    - kmeans_models (list): List of KMeans clustering models for each stratum.
    - iter_first (int): Starting iteration for data collection.
    - iter_last (int): Ending iteration for data collection.
    - bin_boundaries (list): List of boundaries for stratification based on progress coordinate.
    - dimreduce_model: Trained dimensionality reduction model for transforming coordinates.

    Returns:
    - flux_mtx_list (list): List of flux matrices calculated from each west.h5 file / weighted ensemble replicate.
    """
    # List of flux matrices 
    # Each element contains flux matrix calculated from single h5 file / WE replicate 
    flux_mtx_list = []
    
    # Total number of clusters
    n_clusters = len(cluster_centers)
    
    # Total number of iterations used 
    total_iter = iter_last - iter_first + 1
  
    # Loop through h5 files 
    for h5_file in h5_file_paths:
        print(f'Processing {h5_file}')
        
        # Define flux matrix for this replicate
        flux_mtx_replicate = np.zeros((n_clusters, n_clusters))
        
        # Reading the h5 file directly
        with h5py.File(h5_file, "r") as data_file:
            # Loop through the iterations
            for iteration in tqdm(range(iter_first,iter_last+1)):
                # Get initial and final pcoord and coordinates (ones in h5 are full atomic coordinates)
                pcoord_from = data_file['iterations'][f'iter_{iteration:08d}']['pcoord'][:][:,0,:]
                pcoord_to   = data_file['iterations'][f'iter_{iteration:08d}']['pcoord'][:][:,1,:]
                coord_from  = data_file['iterations'][f'iter_{iteration:08d}']['auxdata']['coord'][:][:,0,:,:]
                coord_to    = data_file['iterations'][f'iter_{iteration:08d}']['auxdata']['coord'][:][:,1,:,:]

                # Weights of these trajectories
                full_weight_data = data_file['iterations'][f'iter_{iteration:08d}']['seg_index'][:]
                weight = [item[0] for item in full_weight_data]
                
                # Number of trajectories (or transitions) in this iteration 
                n_traj = np.shape(pcoord_from)[0]
                
                # Define discrete cluster index of these trajectory from and to points
                disc_from = np.zeros(n_traj, dtype='int32')
                disc_to = np.zeros(n_traj, dtype='int32')
                
                # Loop within the trajectories of this iteration 
                for trajectory in tqdm(range(n_traj)):

                    # Get the combined pcoord and dimreduced coord for from and to info 
                    X_from = np.concatenate((pcoord_from[trajectory],
                                             dimreduce_model.transform(intermediate_dimreduce(coord_from[trajectory])[0])))
                    X_to   = np.concatenate((pcoord_to[trajectory],
                                             dimreduce_model.transform(intermediate_dimreduce(coord_to[trajectory])[0])))
               
                    # Assign cluster indices
                    disc_from[trajectory] = stratified_clustering(bin_boundaries, kmeans_models, X_from)
                    disc_to[trajectory] = stratified_clustering(bin_boundaries, kmeans_models, X_to)
                
                # Add the transition info to `flux_mtx_replicate`
                flux_mtx_replicate += coo_matrix(
                    (weight, (disc_from, disc_to)),
                    shape=(n_clusters, n_clusters),
                ).toarray()
                
        # Normalize wrt to total number of iterations
        flux_mtx_replicate /= total_iter
            
        # Add to the list
        flux_mtx_list += [flux_mtx_replicate]
    
    # Return the list of flux matrices 
    return flux_mtx_list

def stratified_clustering(
    bin_boundaries,
    kmeans_models,
    X  # This should be a list with two elements: progress coordinate (pcoord) and dimensionally reduced coordinates (dimreduce)
):
    """
    Perform stratified clustering based on progress coordinate and return the predicted cluster index.

    Args:
    - bin_boundaries (list): A list of boundaries for stratification based on progress coordinate.
    - kmeans_models (list): List of KMeans clustering models for each stratum.
    - X (list): List with two elements - progress coordinate (pcoord) and dimensionally reduced coordinates (dimreduce). Note that it should be concatenated and be a single array. First element will be pcoord (if you're using 1D pcoord) and the rest dimreduce. 

    Returns:
    - int: Predicted cluster index based on stratified clustering.
    """
    # Number of bins
    n_bin = len(bin_boundaries) - 1
    # Offset for the cluster indices in different bins 
    offset = 0
    # Loop through the bins
    for i in range(n_bin):
        # If X is in bin i, return the predicted cluster index
        if bin_boundaries[i] < X[0] and X[0] <= bin_boundaries[i + 1]:
            #print('predicted discretization is: ',kmeans_models[i].predict([X])[0])
            return int(kmeans_models[i].predict([X])[0] + offset)
        # Add offset value based on the number of unique cluster labels in the current stratum
        offset += len(np.unique(kmeans_models[i].labels_))

def get_Tmatrix(
    fluxmatrix_og,
    target_cluster_index,
    basis_cluster_index
):
    """
    Compute the transition matrix from the flux matrix.
    Corrects the "target" states to be true sink states.

    Parameters:
    fluxmatrix_og (np.ndarray): The original flux matrix.
    target_cluster_index (int): Index of the target cluster 
    basis_cluster_index (int) : Index of the basis cluster

    Returns:
    np.ndarray: The transition matrix.

    This function computes the transition matrix from the flux matrix and corrects the "target" states to be true sink states. It normalizes the flux matrix, sets states with 0 flux as self-transitions, and sets target bins to uniformly recycle into basis bins.
    """
    # Get a copy of the flux matrix
    fluxmatrix = fluxmatrix_og.copy()
    # Get the dimension of the flux matrix
    fluxmatrix_shape = np.shape(fluxmatrix_og)
    # Add up the total flux on each row, i.e. from each state
    fluxes_out = np.sum(fluxmatrix_og, axis=1)
    # For each state
    for state_idx in range(fluxmatrix_shape[0]):
        # For positive definite flux, set the matrix elements based on normalized fluxes
        if fluxes_out[state_idx] > 0:
            fluxmatrix[state_idx, :] = fluxmatrix[state_idx, :] / fluxes_out[state_idx]
        # If the flux is zero, then consider it all self-transition
        if fluxes_out[state_idx] == 0.0:
            fluxmatrix[state_idx, state_idx] = 1.0
    # Create a transition matrix
    tmatrix = fluxmatrix.copy()
    tmatrix[target_cluster_index, :] = np.zeros(len(fluxmatrix))
    tmatrix[target_cluster_index, basis_cluster_index] = 1.0

    return tmatrix

def solve_discrepancy(tmatrix, pi, B):
# https://github.com/jdrusso/msm_we/blob/7e346567d246a718d5f8eeff08aa2a5cc3d04b68/msm_we/optimization.py#L15C1-L76C33
    r"""
    Given a transition matrix, solves for the discrepancy function.

    The Poisson equation for the discrepancy function is

    .. math::

        (I - K)h = 1_B - \pi(B), \:\: h \cdot \pi = 0

    however, since :math:`I-K` is singular, we instead solve

    .. math::

        (I - K + \pi \pi^T / || \pi ||^2_2)h = 1_B - \pi(B), \:\: h \cdot \pi = 0

    where :math:`h` is a volumn vector, `1_B` is an indicator function which is 1 in B and 0 everywhere
    else, :math:`\pi` is the steady-state solution of :math:`K`, and `\pi(B)` is a column vector with
    the steady-state value of :math:`\pi(B)` in every element.

    Parameters
    ----------
    tmatrix: 2D array-like,
        Transition matrix
    pi: array-like,
        Steady-state distribution for the input transition matrix
    B: array-like,
        Indices of target states B

    Returns
    --------
    (discrepancy, variance)
    """

    print("Computing pi matrix")
    norm = np.dot(pi, pi.T)

    pi_matrix = pi * pi.T.reshape(-1, 1) / norm

    b_indicator = np.zeros_like(pi)
    b_indicator[B] = 1.0

    pi_b = np.ones_like(pi)
    pi_b[:] = sum(pi[B])

    discrepancy = np.linalg.solve(
        np.identity(tmatrix.shape[0]) - tmatrix + pi_matrix, b_indicator - pi_b
    )

    variance = np.sqrt(
        np.dot(tmatrix, discrepancy**2) - np.dot(tmatrix, discrepancy) ** 2
    )

    if np.isnan(variance).any():
        print("NaN elements in variance!")

    # TODO: Verify this is the correct sanity check
    assert np.isclose(
        discrepancy @ pi, 0
    ), "Discrepancy solution failed normalization sanity check!"

    return discrepancy, variance

def haMSM_result_aggregated(
    haMSM_paths,
    pcoord_bounds
):
    # Read in flux matrix 
    flux_matrices = []
    
    for path in haMSM_paths:
        with open(path+'/flux_matrices.pkl', 'rb') as file:
        # Load the data from the pickle file
            flux_matrix = pickle.load(file)[0]
        flux_matrices += [flux_matrix]
    
    average_flux_matrix = np.mean(flux_matrices, axis=0)
    
    # Cluster center info
    # They should all be the same
    cluster_center = np.loadtxt(haMSM_paths[0]+'/cluster_centers.txt')

    # Getting target and source cluster center
    target_cluster_index = np.where((cluster_center>pcoord_bounds['target'][0][0]) & 
                                    (cluster_center<pcoord_bounds['target'][0][1]))[0][0]
    basis_cluster_index = np.where((cluster_center>pcoord_bounds['basis'][0][0]) & 
                                   (cluster_center<pcoord_bounds['basis'][0][1]))[0][0]

    # Calculate corresponding transition matrix
    transition_matrix = get_Tmatrix(average_flux_matrix,
                                    target_cluster_index, 
                                    basis_cluster_index)
    # Steady state
    steady_state = MarkovStateModel(transition_matrix).stationary_distribution
    
    # Get discrepancy and variance 
    from msm_we import optimization as mo
    discrepancy, variance = solve_discrepancy(transition_matrix, 
                                                 steady_state, 
                                                 [target_cluster_index])

    return cluster_center, steady_state, discrepancy, variance
