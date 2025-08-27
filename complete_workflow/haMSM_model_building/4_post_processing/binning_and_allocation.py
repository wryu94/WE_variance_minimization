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

