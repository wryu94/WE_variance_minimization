import numpy as np
import socket, pickle
import mdtraj as md

print("Starting client script")

HOST, PORT = "127.0.0.1", 5000

def send_request(pcoord, position):
    msg = {"pcoord": pcoord, "position": position}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(pickle.dumps(msg))
        data = s.recv(1000000)
        return pickle.loads(data)

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

# Parameter file, this will be constant for the NTL9 system
WE_ref_file     = "/home/groups/ZuckermanLab/ryuwo/NTL9_WE_benchmark/ntl9_WE_optimized/low_friction_gamma_5_inv_ps_tau_100_ps_pair_dist_featurization/rep_optimization_test/ref_files/"
param_file  = WE_ref_file + 'ntl9.prmtop'
# Restart file, eventually turn this into input 
restart_file = WE_ref_file + 'ntl9.rst7'

# Read in RMSD value from cpptraj output file `rmsd.temp`
pcoord_1 = np.loadtxt('rmsd.temp')[1]

# Get system coordinate u_pos
u_pos_1 = mdtraj_load_pos(restart_file, param_file)

# Get bin assignment
bin_assignment_1 = send_request(pcoord_1, u_pos_1)

# Save as pcoord_and_bin
np.savetxt('pcoord_and_bin.dat', np.column_stack((pcoord_1, bin_assignment_1)), fmt='%.6f %d')
