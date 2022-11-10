import numpy as np
import random
import sys, os
sys.path.append(os.path.abspath('../../lib'))

from data_simulator_full_auto import generate_simulated_data

n_sim_reps = 10

# n_muts = [10, 20, 50, 100]
# n_cells = [100, 500, 1000]
n_muts = [150, 200]
n_cells = [500, 1000]
FPR = 0.001
ADO = 0.3
cell_alpha = 1
mut_alpha = 1
d_rng_id = 1

save_dir = 'sim_data'

for sim in range(n_sim_reps):
    seed = 1000 + sim
    np.random.seed(seed)
    random.seed(seed)
    for n_mut in n_muts:
        n_clust = n_mut
        for n_cell in n_cells:
            data, true_tree = generate_simulated_data(n_clust, 
                                                n_cell, 
                                                n_mut, 
                                                FPR, 
                                                ADO, 
                                                cell_alpha, 
                                                mut_alpha,
                                                d_rng_id
                                                )
            real_data, adj_mat, cell_assignments, mut_assignments = true_tree
            fn = "m{}_c{}_fp{}_ad{}_ca{}_ma{}_dr{}_seed{}".format(n_mut,n_cell,FPR,ADO,cell_alpha,mut_alpha,d_rng_id,seed)
            np.savetxt(os.path.join(save_dir,fn+".data"), data, fmt="%d")
            np.savetxt(os.path.join(save_dir,fn+".dataT"), data.T, fmt="%d")
            np.savetxt(os.path.join(save_dir,fn+".nonoise"), real_data, fmt="%d")
            np.savetxt(os.path.join(save_dir,fn+".adj"), adj_mat, fmt="%d")
            np.savetxt(os.path.join(save_dir,fn+".cell_ass"), cell_assignments, fmt="%d")
            np.savetxt(os.path.join(save_dir,fn+".mut_ass"), mut_assignments, fmt="%d")