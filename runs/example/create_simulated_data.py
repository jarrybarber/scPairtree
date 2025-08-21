import sys, os
import numpy as np
import random
ex_dir = os.path.dirname(os.path.abspath(__file__))
scp_dir = os.path.abspath(os.path.join(ex_dir,"../.."))
sys.path.append(os.path.abspath(os.path.join(scp_dir,"lib")))
sys.path.append(os.path.abspath(os.path.join(scp_dir,"bin")))
from data_simulator import generate_simulated_data
import tree_plotter


def main():
    '''
    Generates some simulated data for the example.
    The parameters below can be changed to produce different sized datasets
    '''
    
    #Dataset parameters
    n_clust = 50
    n_mut = 50
    n_cell = 200
    fpr = 0.0001
    adr = 0.2
    seed = 1000
    d_rng_i = 2

    np.random.seed(seed)
    random.seed(seed)

    data, true_tree = generate_simulated_data(n_clusts=n_clust,n_muts=n_mut,n_cells=n_cell,FPR=fpr,ADO=adr,cell_alpha=1,mut_alpha=1,drange=d_rng_i)
    noise_free_data  = true_tree[0]
    adj_mat          = true_tree[1]
    cell_assignments = true_tree[2]
    mut_assignments  = true_tree[3]

    #For the sake of clarity, sort mutations and cells by their cluster assignments
    mut_sort_inds = np.argsort(mut_assignments)
    mut_assignments = mut_assignments[mut_sort_inds]
    data = data[mut_sort_inds,:]
    noise_free_data = noise_free_data[mut_sort_inds,:]

    cell_sort_inds = np.argsort(cell_assignments)
    cell_assignments = cell_assignments[cell_sort_inds]
    data = data[:,cell_sort_inds]
    noise_free_data = noise_free_data[:,cell_sort_inds]

    #Output simulated data.
    fn = os.path.join(ex_dir, "data", "M{}_N{}_K{}_fpr{}_adr{}_seed{}".format(n_mut, n_cell, n_clust, fpr, adr, seed))
    np.savetxt(fn+".data", data, "%d")
    np.savetxt(fn+".noise_free_data", noise_free_data, "%d")
    np.savetxt(fn+".actual_adj_mat", adj_mat, "%d")
    np.savetxt(fn+".cluster_assignments_muts", mut_assignments, "%d")
    np.savetxt(fn+".cluster_assignments_cells", cell_assignments, "%d")

    fig = tree_plotter.plot_tree(adj_mat,mut_assignments,"Actual Tree")
    fig.savefig(fn+"_actual_tree.png")


if __name__ == "__main__":
    main()