import numpy as np
import pickle, socket
from haMSM_functions import get_Tmatrix, solve_discrepancy, haMSM_result_aggregated, featurization, stratified_clustering
from deeptime.markov.msm import MarkovStateModel
from joblib import load
import mdtraj as md

print("Starting server script")

# Bin boundaries from unoptimized WE
bin_boundaries = [0.0,1.00] + [1.10+0.1*i for i in range(35)] + [4.60+0.2*i for i in range(10)] + [6.60+0.6*i for i in range(6)] + [float('inf')]

# Dimreduce and kmeans model path
# Also read in cluster vs bin index results here
haMSM_path = "/home/groups/ZuckermanLab/ryuwo/NTL9_WE_benchmark/ntl9_haMSM_models/low_friction/pair_featurization/"
dimreduce_path = haMSM_path + '1_train_dimreduce/dimreduce_model.joblib'
kmeans_path = haMSM_path + '3_flux_matrix_calc_save_all_flux_mtx/kmeans_models.joblib'
cluster_vs_binning_path = haMSM_path+"4_post_processing/cluster_vs_binning.npy"
# Load models
dimreduce_model = load(dimreduce_path)
kmeans_models   = load(kmeans_path)
binning         = np.load(cluster_vs_binning_path)

# Define function
def optimized_bin_assignment(
    pcoord,
    position
):
    feature            = featurization(position)
    dimreduce          = dimreduce_model.transform(feature)
    X                  = np.concatenate(([pcoord], dimreduce))
    cluster_assignment = stratified_clustering(bin_boundaries, kmeans_models, X)
    bin_assignment     = binning[cluster_assignment]

    return bin_assignment

# Server
HOST, PORT = "127.0.0.1", 5000
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Server running on {HOST}:{PORT}")
    while True:
        conn, addr = s.accept()
        with conn:
            data = conn.recv(1000000)  # up to ~1 MB
            if not data:
                continue
            msg = pickle.loads(data)
            if msg == "SHUTDOWN":
                print("Server shutting down...")
                break
            try:
                # msg expected to be dict
                pcoord = msg["pcoord"]
                position = msg["position"]
                result = optimized_bin_assignment(pcoord, position)
                conn.sendall(pickle.dumps(result))
            except Exception as e:
                conn.sendall(pickle.dumps({"error": str(e)}))
