import numpy as np
from tree_util import convert_adjmatrix_to_ancmatrix
from pairs_tensor_util import p_data_given_truth_and_errors
from common import DataRange, DataRangeIdx


def _apply_errors(real_data,FPR,ADR):
    # Here, FPR corresponds to the false positive rate on one of the alleles of a locus.
    # This could mean that two false positives could occur, and still get just the one FP
    # Additionally, ADR corresponds to just one allele dropping out. Doesn't necessarily mean
    # a false negative will occur. Also, assuming a diploid cell, the full locus dropout 
    # will be ADR^2

    #Start off with most complex data, then reduce complexity if user asks for it.
    d_rng_i = DataRangeIdx.ref_hetvar_homvar_nodata
    r_rng = DataRange[d_rng_i] #[0,1,2,3]

    data = np.zeros(real_data.shape, dtype=int)
    ps_gt0 = [p_data_given_truth_and_errors(d,0,FPR,ADR,d_rng_i) for d in r_rng]
    ps_gt1 = [p_data_given_truth_and_errors(d,1,FPR,ADR,d_rng_i) for d in r_rng]
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

def _assign_to_subclones(n_assignments, n_clust, min_assigned_per_sc, a=1):
    assert n_assignments >= min_assigned_per_sc*n_clust
    #Determine assignments
    clust_weights = np.random.dirichlet([a]*n_clust)
    assignments = np.random.choice(n_clust, size=n_assignments, p=clust_weights)+1 #+1 to account for added root node
    #Impose the min_assigned criteria by overwriting random indices
    hardset_assignment_inds = np.random.permutation(n_assignments)[0:n_clust*min_assigned_per_sc]
    hardset_assignments = [i+1 for i in range(n_clust) for j in range(min_assigned_per_sc)] #+1 to account for added root node
    assignments[hardset_assignment_inds] = hardset_assignments
    
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

    anc_mat = convert_adjmatrix_to_ancmatrix(adj_mat)

    return adj_mat, anc_mat

def generate_simulated_data(n_clust, n_cell, n_mut, FPR, ADR, cell_alpha, mut_alpha, drange, min_cell_per_node=1, min_mut_per_node=1):
    #These correspond to how the nodes / subclones are related to each other.
    adj_mat, anc_mat = _generate_tree_structure(n_clust)
    
    cell_assignments = _assign_to_subclones(n_cell, n_clust, min_cell_per_node, a=cell_alpha)
    mut_assignments  = _assign_to_subclones(n_mut,  n_clust, min_mut_per_node, a=mut_alpha)

    real_data = _generate_error_free_data(anc_mat,cell_assignments,mut_assignments)
    data = _apply_errors(real_data,FPR,ADR)
    data = _put_data_in_drange_format(data,drange)
    return data, (real_data, adj_mat, cell_assignments, mut_assignments)


def main():
    print("data_simulator is not callable by itself")
    pass


if __name__ == "__main__":
    main()