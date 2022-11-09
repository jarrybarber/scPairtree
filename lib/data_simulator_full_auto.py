import numpy as np
from scipy.linalg import block_diag
import random
import os
import sys
import argparse
from util import DATA_DIR
from tree_util import make_ancestral_from_adj
from pairs_tensor_util import p_data_given_truth_and_errors
from common import DataRange, DataRangeIdx


def _save_data(data,real_tree_info):
    #Can do this one later when I feel like it.
    #Will have to set an output location and save all of the input parameters as well.
    return

def _apply_errors(real_data,FPR,ADO):
    # Here, FPR corresponds to the false positive rate on one of the alleles of a locus.
    # This could mean that two false positives could occur, and still get just the one FP
    # Additionally, ADO correspongs to just one allele dropping out. Doesn't necessarily mean
    # a false negative will occure. Also, assuming a diploid cell, the full locus dropout 
    # will be ADO^2

    #Set what the data points can be
    d_rng_i = DataRangeIdx.ref_hetvar_homvar_nodata #Just leave as this? Later on can just merge 3s into 0s and 2s into 1s if really want to use the other types, and then just have to make the data once.
    r_rng = DataRange[d_rng_i] #[0,1,2,3]

    data = np.zeros(real_data.shape, dtype=int)
    ps_gt0 = [p_data_given_truth_and_errors(d,0,FPR,ADO,d_rng_i) for d in r_rng]
    ps_gt1 = [p_data_given_truth_and_errors(d,1,FPR,ADO,d_rng_i) for d in r_rng]
    data = data + np.multiply((real_data==0).astype(int), np.random.choice(r_rng, data.shape, p=ps_gt0))
    data = data + np.multiply((real_data==1).astype(int), np.random.choice(r_rng, data.shape, p=ps_gt1))

    return data

def _put_data_in_drange_format(data, d_rng_id=DataRangeIdx.ref_var_nodata):
    if d_rng_id==DataRangeIdx.ref_hetvar_homvar_nodata:
        return np.copy(data)
    elif d_rng_id==DataRangeIdx.ref_var_nodata:
        to_ret = np.copy(data)
        to_ret[to_ret==2] = 1
        return to_ret
    elif d_rng_id==DataRangeIdx.var_notvar:
        to_ret = np.copy(data)
        to_ret[to_ret==2] = 1
        to_ret[to_ret==3] = 0
        return to_ret

def _generate_error_free_data(anc_mat, cell_assignments, mut_assignments):
    #Switch to one-hot encoding
    CA = np.eye(anc_mat.shape[0])[cell_assignments].T
    MA = np.eye(anc_mat.shape[0])[mut_assignments].T

    error_free_dat = MA.T @ anc_mat @ CA
    return error_free_dat.astype(int)

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
    adj_mat = np.eye(n_clust+1, dtype=int)
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

def generate_simulated_data(n_clust,n_cells,n_muts,FPR,ADO,cell_alpha,mut_alpha,drange):
    adj_mat, anc_mat = _generate_tree_structure(n_clust)
    
    cell_assignments = _assign_to_subclones(n_cells, n_clust, a=cell_alpha)
    mut_assignments  = _assign_to_subclones(n_muts,  n_clust, a=mut_alpha)

    real_data = _generate_error_free_data(anc_mat,cell_assignments,mut_assignments)
    data = _apply_errors(real_data,FPR,ADO)
    data = _put_data_in_drange_format(data,drange)
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
    parser.add_argument('--data-range', dest='d_rng_id', type=int, default=1,
        help='Data range id. There are 3 options: (0: [0,1]; 1: [0,1,3]; 2: [1,2,3])')
    parser.add_argument('--cell-alpha', dest='cell_alpha', type=float, default=0.5,
        help='Dirichlet distribution parameter for distributing cells to the clusters.')
    parser.add_argument('--mut-alpha', dest='mut_alpha', type=float, default=1.,
        help='Dirichlet distribution parameter for distributing mutations to the clusters.')
    parser.add_argument('--sim-isav', dest='ISAs', action='store_true',
        help='Whether or not to simualate ISA violations.')
    parser.add_argument('--save-data', dest='save_data', action='store_true',
        help='Whether or not to save the data. If true nothing will be returned.')

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
        args.mut_alpha,
        args.d_rng_id
        )

    if args.save_data:
        _save_data(data,real_tree_info)
        return
    else:
        return data




if __name__ == "__main__":
    main()