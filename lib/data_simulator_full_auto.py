import numpy as np
from scipy.linalg import block_diag
import random
import os
import sys
import argparse
from util import DATA_DIR
from tree_util import make_ancestral_from_adj


def _save_data(data):
    #Can do this one later when I feel like it.
    #Will have to set an output location and save all of the input parameters as well.
    return

def _apply_errors(real_data,FPR,ADO):
    # Here, FPR corresponds to the false positive rate on one of the alleles of a locus.
    # This could mean that two false positives could occur, and still get just the one FP
    # Additionally, ADO correspongs to just one allele dropping out. Doesn't necessarily mean
    # a false negative will occure. Also, assuming a diploid cell, the full locus dropout 
    # will be ADO^2
    n_snvs, n_cells = real_data.shape

    locus_DOR = ADO**2
    actual_FPR = FPR**2*(1-ADO)**2 + 2*FPR*(1-FPR)*(1-ADO)**2 + 2*FPR*ADO*(1-ADO)
    actual_FNR = 2*FPR*(1-FPR)*(1-ADO)**2 + ADO*(1-ADO)

    data = np.copy(real_data)
    data[np.random.rand(n_snvs,n_cells)<=locus_DOR] = 3

    FN_inds = (data==1) & (np.random.rand(n_snvs,n_cells)<=actual_FNR)
    FP_inds = (data==0) & (np.random.rand(n_snvs,n_cells)<=actual_FPR)
    data[FN_inds] = 0
    data[FP_inds] = 1

    return data, (actual_FPR*(1-locus_DOR), actual_FNR*(1-locus_DOR), locus_DOR)

def _generate_error_free_data(anc_mat, cell_assignments, mut_assignments):
    #Switch to one-hot encoding
    CA = np.eye(anc_mat.shape[0])[cell_assignments].T
    MA = np.eye(anc_mat.shape[0])[mut_assignments].T

    error_free_dat = MA.T @ anc_mat @ CA
    return error_free_dat

def _assign_to_subclones(n_assignments, n_clust, a=1):
    assert n_assignments >= n_clust
    clust_weights = np.random.dirichlet([a]*n_clust)
    assignments = np.append(
                    [i for i in range(1,n_clust+1)],
                    np.random.choice(n_clust, size=(n_assignments-n_clust), p=clust_weights)+1
    )
    return assignments

def _generate_tree_structure(n_clust):
    #All nodes are considered adjacent to themselves. Add one more cluster for the root node.
    adj_mat = np.eye(n_clust+1)
    #Set the first node to branch off the root node
    adj_mat[0,1] = 1
    #Iterate through the rest of the nodes. Either continue a chain with prob 3/4 or start a new chain with prob 1/4.
    for node in range(2,n_clust+1):
        if np.random.rand() <= 0.75:
            adj_mat[node-1,node] = 1
        else:
            new_parent = np.random.randint(0,node)
            adj_mat[new_parent,node] = 1

    anc_mat = make_ancestral_from_adj(adj_mat)

    return adj_mat, anc_mat

def generate_simulated_data(n_clust,n_cells,n_muts,FPR,ADO,cell_alpha,mut_alpha):
    adj_mat, anc_mat = _generate_tree_structure(n_clust)
    
    cell_assignments = _assign_to_subclones(n_cells, n_clust, a=cell_alpha)
    mut_assignments  = _assign_to_subclones(n_muts,  n_clust, a=mut_alpha)

    real_data = _generate_error_free_data(anc_mat,cell_assignments,mut_assignments)
    data, _ = _apply_errors(real_data,FPR,ADO)
    return data, (real_data, adj_mat, cell_assignments, mut_assignments)

def get_args():
    parser = argparse.ArgumentParser(
        description='Simulate single-cell tumour data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--seed', dest='seed', type=int, default=np.random.randint(2**32),
        help='Integer seed used for pseudo-random number generator.')
    parser.add_argument('--name', dest='name', type=str, default=None,
        help='(Optional) The name of the tree being created')
    parser.add_argument('K', dest='n_clust', type=int, default=30,
        help='Number of subclones to simulate.')
    parser.add_argument('C', dest='n_cells', type=int, default=10,
        help='Number of cells per subclone.')
    parser.add_argument('M', dest='n_muts', type=int, default=10,
        help='Number of mutations per subclone.')
    parser.add_argument('A', dest='ADO', type=float, default=0.5,
        help='Allelic dropout rate.')
    parser.add_argument('P', dest='FPR', type=float, default=0.005,
        help='False positive rate.')
    parser.add_argument('--cell-alpha', dest='cell_alpha', type=float, default=0.5,
        help='Dirichlet distribution parameter for distributing cells to the clusters.')
    parser.add_argument('--mut-alpha', dest='mut_alpha', type=float, default=1.,
        help='Dirichlet distribution parameter for distributing mutations to the clusters.')
    parser.add_argument('--sim-isav', dest='ISAs', action='store_true',
        help='Whether or not to simualate ISA violations.')
    parser.add_argument('--save-data', dest='save_data', action='store_true',
        help='Whether or not to simualate save the data. If true nothing will be returned.')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    data, real_tree_info = generate_simulated_data(
        args.n_clust,
        args.n_cells,
        args.n_muts,
        args.FPR,
        args.ADO,
        args.cell_alpha,
        args.mut_alpha
        )

    if args.save_data:
        _save_data(data)
        return
    else:
        return data




if __name__ == "__main__":
    main()