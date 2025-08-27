import numpy as np
from joblib import load
from binning_and_allocation import get_uniform_pi_v_bins

# Bin boundaries from unoptimized WE
bin_boundaries = [0.0,1.00] + [1.10+0.1*i for i in range(35)] + [4.60+0.2*i for i in range(10)] + [6.60+0.6*i for i in range(6)] + [float('inf')]
n_bins = len(bin_boundaries)-1

# Read in haMSM functions 
# [0]: cluster center pcoord
# [1]: steady state
# [2]: discrepancy
# [3]: variance 
haMSM_results = load("./haMSM_results.joblib")

# Get binning 
# [1:-1] to only get the active clusters that aren't in basis or target 
binning = get_uniform_pi_v_bins(haMSM_results[3][1:-1],
                                haMSM_results[2][1:-1],
                                haMSM_results[1][1:-1],
                                n_bins)
# Kinda wack, but it works: because we want to have all possible assinment of bins, we want to include the results for basis and target clusters. The details of this modification will depend on the system 
# By convention, 
# basis cluster (last) is assigned 0th bin 
# target cluster (0th) is assigned (n_bin-1)th bin 
binning = np.concatenate(([n_bins-1], binning+1, [0]))

# SAve and test
np.save('cluster_vs_binning',binning)
test = np.load('cluster_vs_binning.npy',allow_pickle=True)

print(binning)
print(test)
print(np.array_equal(binning,test))
