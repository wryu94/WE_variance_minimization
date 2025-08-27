import h5py
import numpy as np
import pickle
import sys
import deeptime
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from sklearn.cluster import KMeans
from msm_we import optimization as mo
from deeptime.markov.msm import MarkovStateModel, MaximumLikelihoodMSM
import mdtraj as md
from scipy.sparse import coo_matrix
from MDAnalysis.analysis import rms
from tqdm import tqdm
from deeptime.decomposition import TICA, VAMP

# Reference structure
ref_file = 'reference.pdb'
import MDAnalysis as mda
import numpy as np

def intermediate_dimreduce(coords):
    """
    This function defines the intermediate-ly reduced coordinate (opposed to the full 
    atomic coordinate).

    Choice / detail of this function should depend on the system of interest. Since
    I'm working on protein folding/unfolding, this is a pairwise-distance function 
    that calculates the aligned CA distances of the current structure (coords) against
    the reference structure (ref_file).

    Calculates distances between atoms in the backbone of a molecular structure 
    for each frame in the provided coordinates.

    Args:
    - coords: File or data containing molecular coordinates.

    Returns:
    - dist_out: NumPy array containing distances calculated for each frame.
    """
    # Initialize MDAnalysis Universe objects using the reference file
    u_ref = mda.Universe(ref_file)
    u_check = mda.Universe(ref_file)
    # List to store calculated distances
    dist_out = []
    # Load new coordinates into the u_check Universe
    u_check.load_new(coords)
    # Iterate through frames in the trajectory of u_check
    for frame in u_check.trajectory:
        # Calculate distances between atoms in the backbone of u_check and u_ref
        dists = distances.dist(
            u_check.select_atoms('name CA'),#'backbone'),
            u_ref.select_atoms('name CA')#'backbone')
        )[2]
        # Append calculated distances to dist_out
        dist_out.append(dists)
    # Convert dist_out into a NumPy array
    dist_out = np.array(dist_out)
    # Return the distances
    return dist_out

def processCoordinates(coords):
# Here coord is np array of the structure, in the same format as pdb
# I've checked that this gives same answer as what you get from cpptraj
# Calculates RMSD of `coords` wrt the reference structure in `reference.pdb`
    # Load reference structure in mdtraj
    reference = md.load(ref_file)
    # Get alpha carbon indices
    alpha_indices = reference.top.select_atom_indices('alpha')
    # Make mdtraj object from the coord
    check = md.Trajectory(coords, reference.top)
    # Calculate rmsd (alpha carbon) 
    # 10 for unit conversion (angstrom to nm?)
    rmsd = md.rmsd(reference, check, atom_indices=alpha_indices) * 10
    # Return result
    return rmsd

class IdentityTransformModel:
    def transform(coord):
        return coord

def train_dimreduce_start(
    dimreduce_method, # 'vamp', 'tica', or 'none' for now
    h5_file_paths,
    iter_first,
    iter_last
):
# Return the dimreduce object trained on iteration data from h5
# Note that since neither VAMP or TICA uses trajectory weight, this doesn't fully
# use the WE data, but seems like it's the best we can do for now
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
    # model loaded is what to continue training on 

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
    Directly access the west.h5 files and collect the WE data (pcoord and dimreduce (calculated from coord (pairwise CA distance and dimreduce on that using the model that we trained))
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

    Parameters:
    bin_boundaries (list): A list of boundaries for stratification based on progress coordinate.
    clusters_per_bin (list): The number of clusters to create per bin.
    WE_data_concat (dict): A dictionary containing concatenated weighted ensemble data.

    Returns:
    list: A list of KMeans clustering models for each stratum, with the first and last strata having a single cluster.
    np.ndarray: An array of cluster centers.
    
    This function performs stratified clustering based on progress coordinate and returns a list of KMeans models, one for each stratum. The number of clusters is specified for each stratum, except for the first and last strata where only one cluster is created. It also returns an array of cluster centers.
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
            # Updated: in ntl9 case, key = pcoord (RMSD) and item is dict with 'coord' and 'weight'
            if bin_boundaries[i] < item['pcoord'] and item['pcoord'] < bin_boundaries[i + 1]:
                data_point_in_this_bin[counter] = item
                counter += 1

        # Now we want to get X (pcoord and dimreduce) and weight
        #X = []
        #weight = []
        #for key, item in data_point_in_this_bin.items():
        #    X += [np.concatenate(([item['pcoord']], item['dimreduce']))]
        #    weight += [item['weight']]

        # Now we want to get X (pcoord and dimreduce) and weight
        # This probably won't be necessary for AA WE runs, but may need it for synD
        # Essentially we're concatenating multiple same structures that was accounted for 
        # in the concatenated WE data
        X = []
        weight = []
        for key, item in data_point_in_this_bin.items():
            array_i = np.concatenate(([item['pcoord']], item['dimreduce']))
            weight_i = item['weight']

            if X and np.any(np.all(X == array_i, axis=1)):
            #if np.any(np.isin(array_i, X)):
                #print(np.where(np.all(X==array_i, axis=1)), np.any(np.isin(array_i, X)))
                index = np.where(np.all(X==array_i, axis=1))[0][0]
                weight[index] += weight_i
            else:
                X += [array_i]
                weight += [weight_i]
            #X += [np.concatenate(([item['pcoord']], item['dimreduce']))]
            #weight += [item['weight']]

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
               
                    #print(X_from)
                    #print(X_to) 
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
