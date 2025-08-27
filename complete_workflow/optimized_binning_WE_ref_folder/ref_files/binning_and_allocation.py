from westpa.core.propagators import WESTPropagator
from westpa.core.systems import WESTSystem
from westpa.core.binning import VectorizingFuncBinMapper, RectilinearBinMapper
from sklearn.cluster import KMeans
import numpy as np
from numpy.random import uniform
from numpy import random
from numpy import linspace, int32, float32, array, empty, zeros, int_, any
from numpy.random import Generator, PCG64
from numpy import sqrt as np_sqrt

def get_uniform_pi_v_bins(variance, discrepancy, steady_state, n_desired_we_bins):
    # The last two elements of this are the basis and target states respectively
    pi_v = steady_state * variance
    pi_v_sort = np.argsort(discrepancy).squeeze()
    cumsum = np.cumsum(pi_v[pi_v_sort])

    n_active_bins = n_desired_we_bins - 2
	# minus 2 because we're not counting the basis and target bins 

    bin_bounds = np.linspace(0, cumsum[-1], n_active_bins + 1)[1:]
    bin_assignments = np.digitize(cumsum, bin_bounds, right=True)
    bin_states = bin_assignments[np.argsort(pi_v_sort)]

    return bin_states

"""
def get_uniform_h_bins(variance, discrepancy, steady_state, n_bins):
# modified version of get_uniform_mfpt_bins
    h = discrepancy
    h_sort = np.argsort(discrepancy).squeeze()
    obj_fn = h[h_sort]

    n_active_bins = n_bins - 1

    bin_bounds = np.linspace(0, obj_fn[-1], n_active_bins + 1)[1:]
    bin_assignments = np.digitize(obj_fn, bin_bounds, right=True)
    bin_states = bin_assignments[np.argsort(h_sort)]

    return bin_states

def get_clustered_h_bins(variance, discrepancy, steady_state, n_bins):
# modified version of get_clustered_mfpt_bins
    h = discrepancy
    h_sort = np.argsort(discrepancy).squeeze()
    obj_fn = h[h_sort]

    n_active_bins = n_bins - 1

    clusterer = KMeans(n_clusters=min(n_active_bins, len(obj_fn)))

    we_bin_assignments = clusterer.fit_predict(obj_fn.reshape(-1,1))

    bin_states = np.full_like(obj_fn, fill_value=np.nan)
    for i in range(n_active_bins):
        indices = np.argwhere(we_bin_assignments == i).squeeze()
        states_in_bin = h_sort[indices]
        bin_states[states_in_bin] = i
        print('bin ', i, ' contains ', states_in_bin, ' microstates')

    return bin_states

def get_clustered_pi_v_bins(
    variance, discrepancy, steady_state, n_desired_we_bins, seed=None
):
    # The last two elements of this are the basis and target states respectively
    pi_v = steady_state * variance
    n_active_bins = n_desired_we_bins - 1
    pi_v_sort = np.argsort(discrepancy).squeeze()
    cumsum = np.cumsum(pi_v[pi_v_sort])

    clusterer = KMeans(n_clusters=min(n_active_bins, len(cumsum)), random_state=seed)

    # we_bin_assignments = clusterer.fit_predict(pi_v.reshape(-1, 1))
    we_bin_assignments = clusterer.fit_predict(cumsum.reshape(-1, 1))

    bin_states = np.full_like(cumsum, fill_value=np.nan)
    for i in range(n_active_bins):
        indices = np.argwhere(we_bin_assignments == i).squeeze()
        states_in_bin = pi_v_sort[indices]
        bin_states[states_in_bin] = i

    return bin_states

def get_allocation(binning, discrepancy, obj_fn, n_bins, avg_alloc):
    h_sort = np.argsort(discrepancy).squeeze()

    obj_fn = obj_fn[h_sort]

    n_active_bins = n_bins - 1
    n_walker_intermediate = avg_alloc * n_active_bins

    allocation = np.zeros(int(n_bins))

    for i in range(len(allocation)-1):
        for j in range(len(binning)):
            if binning[j] == i:
                allocation[i] += obj_fn[j] #* n_walker_intermediate

    # Lol added normalization, Nov 10, 2022
    norm = np.sum(allocation)
    allocation *= n_walker_intermediate / norm

    for i in range(len(allocation)-1):
        if round(allocation[i]) == 0:
            allocation[i] = 1
        else:
            allocation[i] = round(allocation[i])

    allocation[-1] = 0
        # Target bin

    print(f"Allocation for each bin is: {allocation}")
    print(f"Total allocation is: {np.sum(allocation)}")

    return allocation
"""
