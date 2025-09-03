Each folder contains all the data / functions / conda environment needed to perform various steps in the WE variance minimization pipeline. 

The schematic of the WE variance optimization is outlined below: 
<img width="6173" height="1883" alt="WE-optimization-scheme" src="https://github.com/user-attachments/assets/7995e973-79d7-46cd-a308-481dfd5ac1f0" />

For details of the project, please see our preprint. To be published at Journal of Chemical Theory and Computation. 
https://arxiv.org/abs/2504.21663

- Unoptimized WE
        - Unoptimized WE of NTL9 as outlined in https://pubs.acs.org/doi/10.1021/jacs.8b10735, only changed resampling time (tau) from 10 ps to 100 ps (ref_files/md-*.in).
        - To run, modify parameters and settings in system.py and west.cfg and simply submit the numbered slurm scripts in order.

- haMSM model building
        - Each folder performs different stages of haMSM model building using WE data. You can technically run everything in one script, but it's divided into smaller steps as these were time intensive calculations.
        - To run, go through the ordered folders, change the file paths in the script.py accordingly and submit the slurm scirpts.

- Optimized WE
        - In this folder, you can run optimized binning WE.
        - Due to technical issues, I set the bin assignment as the second pcoord. The optimal binning is calculated at ref_files/get_pcoord_and_bin_init.py and ref_files/get_pcoord_and_bin_run.py.
        - Original implementation was to launch these optimal bin assignment scripts for each segment after the dynamics steps were run. However, the walltime became too long as you had to repeatedly launch Python environments and read in the haMSM models. As a workaround, I've implemented a server-client solution. The haMSM models and the optimal bin assignment function is stored in the server at server.py. The server is launched before starting the WE runs (as in 1_init_and_run.slurm and 2_continue_run.slurm). Then the ref_files/get_pcoord_and_bin_init.py and ref_files/get_pcoord_and_bin_run.py are the client scripts that uses the function from the server script throughout the WE simulation. While this is not the smoothest implementation, the simulations run a lot faster (as fast as unoptimized, if not slightly faster) as we have removed the environment setup and file IO overhead.
