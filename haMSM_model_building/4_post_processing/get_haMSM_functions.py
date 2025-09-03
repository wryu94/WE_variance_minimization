import numpy as np
import MDAnalysis as mda
import MDAnalysis.analysis.rms as rms
import pickle
from haMSM_functions import get_Tmatrix, solve_discrepancy, haMSM_result_aggregated
from deeptime.markov.msm import MarkovStateModel

# Boundaries of the basis/target, in progress coordinate space
pcoord_bounds = {
    'basis': [[9.6, 100.0]],
    'target': [[-0.01, 1.0]]
}

# Get haMSM_results which contain cluster pcoord, pi, h, and v
haMSM_path = "/home/groups/ZuckermanLab/ryuwo/NTL9_WE_benchmark/ntl9_haMSM_pair_dist_featurization/low_friction/3_flux_matrix_calc_save_all_flux_mtx/"
haMSM_results = haMSM_result_aggregated(haMSM_path, pcoord_bounds) 

# Save haMSM results
with open('haMSM_results.pkl', 'wb') as f:
    pickle.dump(haMSM_results, f)
