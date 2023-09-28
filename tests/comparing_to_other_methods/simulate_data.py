import numpy as np
import random
import sys, os
sys.path.append(os.path.abspath('../../lib'))

from data_simulator import generate_simulated_data
import util

n_sim_reps = 10

#First dataset: how scPairtree does with varying (mut,cell) combos and middling error rates
# n_muts = [10, 20, 50, 100, 200]
# cell_multipliers = [5, 10, 20, 50, 100]
# FPRs = [0.001]
# ADOs = [0.3]
# cell_alpha = 1
# mut_alpha = 1
# d_rng_ids = [1]
# min_cells_per_node = 2

#Second dataset: how scPairtree does with set (mut,cell) and varying error rates
n_muts = [100]
cell_multipliers = [5, 50]
FPRs = [0.001, 0.01, 0.1]
ADOs = [0.1, 0.3, 0.5]
cell_alpha = 1
mut_alpha = 1
d_rng_ids = [1]
min_cells_per_node = 2

save_dir = os.path.join('sims', 'data')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for sim in range(n_sim_reps):
    seed = 1000 + sim
    np.random.seed(seed)
    random.seed(seed)
    for n_mut in n_muts:
        n_clust = n_mut
        for cell_mult in cell_multipliers:
            n_cell = n_mut*cell_mult
            for FPR in FPRs:
                for ADO in ADOs:
                    for d_rng_id in d_rng_ids:
                        fn = "m{}_c{}_fp{}_ad{}_ca{}_ma{}_dr{}_mc{}_seed{}".format(n_mut, n_cell, FPR, ADO, cell_alpha, mut_alpha, d_rng_id, min_cells_per_node, seed)
                        print(fn)
                        if os.path.exists(os.path.join(save_dir,fn+".data")):
                            print("Data already created for file", fn)
                            continue

                        data, true_tree = generate_simulated_data(n_clust, 
                                                            n_cell, 
                                                            n_mut, 
                                                            FPR, 
                                                            ADO, 
                                                            cell_alpha, 
                                                            mut_alpha,
                                                            d_rng_id,
                                                            min_cell_per_node=min_cells_per_node, 
                                                            min_mut_per_node=1
                                                            )
                        real_data, adj_mat, cell_assignments, mut_assignments = true_tree
                        mut_adj_mat = util.convert_nodeadj_to_mutadj(adj_mat,mut_assignments)

                        np.savetxt(os.path.join(save_dir,fn+".data"), data, fmt="%d")
                        np.savetxt(os.path.join(save_dir,fn+".dataT"), data.T, fmt="%d")
                        np.savetxt(os.path.join(save_dir,fn+".nonoise"), real_data, fmt="%d")
                        np.savetxt(os.path.join(save_dir,fn+".adj"), adj_mat, fmt="%d")
                        np.savetxt(os.path.join(save_dir,fn+".cell_ass"), cell_assignments, fmt="%d")
                        np.savetxt(os.path.join(save_dir,fn+".mut_ass"), mut_assignments, fmt="%d")
                        np.savetxt(os.path.join(save_dir,fn+".mut_adj"), mut_adj_mat, fmt="%d")
                    