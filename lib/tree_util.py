import numpy as np
from numba import njit
import common
import util
from pairs_tensor_util import p_data_given_truth_and_errors

@njit(cache=True)
def _get_breadth_first_traversal(adj,start_ind=0):

    bst = [start_ind]
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if bst[i] == j:
                continue
            if adj[bst[i],j] == 1:
                bst.append(j)
    return bst


@njit(cache=True)
def calc_tree_llh(data,anc,mut_ass,fpr,ado,d_rng_i):
    d_range = common.get_d_range(d_rng_i)
    n_mut, n_cell = data.shape
    n_clst = len(np.unique(mut_ass))
    adj = convert_ancmatrix_to_adjmatrix(anc)
    parents = convert_adjmatrix_to_parents(adj)
    node_order = _get_breadth_first_traversal(adj)

    p_dgas = np.zeros((4,2,n_mut))
    for d in d_range:
        for t in (0,1):
            for m in range(n_mut):
                p_dgas[d,t,m] = np.log(p_data_given_truth_and_errors(d,t,fpr[m],ado[m],d_rng_i))
        
    outer_sum = 0
    for i in range(n_cell):
        #Consider the cell being attached to the root first
        j = 0 
        score_cell_at_nodes = np.zeros(n_clst+1)
        for k in range(n_mut):
            p_dga = p_dgas[data[k,i], anc[mut_ass[k],j], k]
            score_cell_at_nodes[j] += p_dga
        #Can now consider the rest of the nodes, making use 
        #of the information calculated for each node's parent.
        for j in node_order[1:]:
            parent = parents[j-1]
            score_cell_at_nodes[j] = score_cell_at_nodes[parent]
            for k in range(n_mut):
                if not (mut_ass[k]==j):
                    continue
                parent_cont_to_sub = p_dgas[data[k,i],0,k]
                this_cont_to_add   = p_dgas[data[k,i],1,k]
                score_cell_at_nodes[j] += this_cont_to_add - parent_cont_to_sub
        score_cell_at_nodes = score_cell_at_nodes[1:]
        B = np.max(score_cell_at_nodes)
        mid = np.log(np.sum(np.exp(score_cell_at_nodes - B))) + B
        outer_sum += mid
    return outer_sum


@njit(cache=True)
def convert_adjmatrix_to_ancmatrix(adj):
    #Note: taken from Pairtree's util code.
    K = len(adj)
    root = 0

    assert np.all(1 == np.diag(adj))
    expected_sum = 2 * np.ones(K)
    expected_sum[root] = 1
    assert np.array_equal(expected_sum, np.sum(adj, axis=0))

    Z = np.copy(adj)
    # np.fill_diagonal(Z, 0)
    for i in range(Z.shape[0]):
        Z[i,i] = 0
        
    stack = [root]
    while len(stack) > 0:
        P = stack.pop()
        C = np.flatnonzero(Z[P])
        if len(C) == 0:
            continue
        # Set ancestors of `C` to those of their parent `P`.
        C_anc = np.copy(Z[:,P])
        C_anc[P] = 1
        # Turn `C_anc` into column vector.
        Z[:,C] = np.expand_dims(C_anc, 1)
        stack += list(C)
    
    for i in range(Z.shape[0]):
        Z[i,i] = 1
    assert np.array_equal(Z[root], np.ones(K))
    
    return Z


@njit(cache=True)
def convert_ancmatrix_to_adjmatrix(anc):
    this_anc = np.copy(anc)
    for i in range(this_anc.shape[0]):
        this_anc[i,i] = 0
    
    adj = np.zeros(this_anc.shape, dtype=np.int8)
    for i in range(adj.shape[0]):
        adj[i,i] = 1
    
    n_clust = len(this_anc)
    for child in range(1,n_clust):
        is_anc_to_child = np.argwhere(this_anc[:,child]).flatten()
        par = 0
        for j in is_anc_to_child:
            is_dec_cur_par = this_anc[par,j]
            if is_dec_cur_par:
                par = j
        adj[par,child] = 1
    return adj


@njit(cache=True)
def convert_ancmatrix_to_parents(anc):
    n_clust = len(anc)
    this_anc = np.copy(anc)
    for i in range(n_clust):
        this_anc[i,i] = 0
    
    parents = np.zeros(n_clust-1, dtype=np.int32)
    for child in range(1,n_clust):
        is_anc_to_child = np.flatnonzero(this_anc[:,child])# np.argwhere(this_anc[:,child]).flatten()
        parent = 0
        for j in is_anc_to_child:
            is_dec_cur_par = this_anc[parent,j]
            if is_dec_cur_par:
                parent = j
        parents[child-1] = parent
    return parents


@njit(cache=True)
def convert_parents_to_ancmatrix(parents):
    adj = convert_parents_to_adjmatrix(parents)
    anc = convert_adjmatrix_to_ancmatrix(adj)
    return anc


#Taken from Pairtree
@njit(cache=True)
def convert_parents_to_adjmatrix(parents):
    K = len(parents) + 1
    adjm = np.eye(K, dtype=np.int8)
    for i,parent in enumerate(parents):
        adjm[parent,i+1] = 1
    return adjm


#Taken from Pairtree
@njit(cache=True)
def convert_adjmatrix_to_parents(adj):
    adj = np.copy(adj)
    np.fill_diagonal(adj, 0)
    parents = np.zeros(adj.shape[1]-1, dtype=np.int32)
    for i in range(1,adj.shape[1]):
        parents[i-1] = util.find_first(1,adj[:,i])
    return parents#np.argmax(adj[:,1:], axis=0)
