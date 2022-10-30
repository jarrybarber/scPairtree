import sys, os
import numpy as np
import numba
import common
import time
import matplotlib.pyplot as plt
import random
import multiprocessing

from common import Models, NUM_MODELS
from util import logsumexp


def _calc_posterior_norm_constant(llhs,samp_probs):
    n_samples = len(llhs)
    nlogC = -np.log(n_samples) + logsumexp(llhs - samp_probs)
    return nlogC

@numba.njit(cache=True)
def _get_anc_vals_from_rel(rel):
    if rel == Models.A_B:
        a,b = 1,0
    elif rel == Models.B_A:
        a,b = 0,1
    elif rel == Models.diff_branches:
        a,b = 0,0
    return a,b


@numba.njit(cache=True)
def _propogate_rules(anc,i,j,rel,model_probs):
    # this will be the tricky part:
    # If set i branched to j, then:
    #  - All nodes desended from i are branched to all nodes desended from j
    #  - All nodes dec(i) cannot be anc to all nodes anc(j)
    #  - All nodes dec(j) cannot be anc to all nodes anc(i)
    # If set i ancestral to j, then:
    #  - All nodes anc(i) will be ancestral to all nodes dec(j)
    #  - All nodes branch(i) will be branched to all nodes dec(j)
    #  - All nodes anc(i) cannot be branched to all nodes anc(j)
    # If set i desendant to j, then equiv to set j ancestral i

    # print(i,j,rel)
    a,b = _get_anc_vals_from_rel(rel)
    anc[i+1,j+1] = a
    anc[j+1,i+1] = b
    model_probs[i,j,:] = -np.inf
    model_probs[j,i,:] = -np.inf
    n_muts = model_probs.shape[0]


    if rel == Models.A_B or rel == Models.B_A:
        if rel == Models.B_A:
            i,j = j,i
        for k in range(n_muts):
            if k in (i,j):
                continue
            if not np.logical_xor(anc[i+1,k+1] == -1, anc[j+1,k+1] == -1):
                continue 

            if anc[i+1,k+1] == 1 and anc[k+1,i+1] == 0: # i anc k
                #j and k can be in any relationship
                continue
            elif anc[i+1,k+1] == 0 and anc[k+1,i+1] == 1: # i dec k
                #j must be dec from k
                _propogate_rules(anc,k,j,Models.A_B,model_probs) 
            elif anc[i+1,k+1] == 0 and anc[k+1,i+1] == 0: # i brn k
                #j must be brn from k
                _propogate_rules(anc,j,k,Models.diff_branches,model_probs) 
            elif anc[j+1,k+1] == 1 and anc[k+1,j+1] == 0: # j anc k
                #i must be anc to k
                _propogate_rules(anc,i,k,Models.A_B,model_probs) 
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 1: # j dec k
                 #i and k must not be branched
                model_probs[i,k,Models.diff_branches]=-np.inf
                model_probs[k,i,Models.diff_branches]=-np.inf
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 0: # j brn k
                #k must not be anc i
                model_probs[i,k,Models.B_A]=-np.inf 
                model_probs[k,i,Models.A_B]=-np.inf
            
            if np.sum(model_probs[i,k,:]==np.NINF)==4:
                rel = np.argwhere(model_probs[i,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,i,k,rel,model_probs)
            if np.sum(model_probs[j,k,:]==np.NINF)==4:
                rel = np.argwhere(model_probs[j,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,j,k,rel,model_probs)
    elif rel == Models.diff_branches:
        for k in range(n_muts):
            if k in (i,j):
                continue
            if not np.logical_xor(anc[i+1,k+1] == -1, anc[j+1,k+1] == -1):
                continue

            if anc[i+1,k+1] == 1 and anc[k+1,i+1] == 0: # i anc k
                #j must be brn from k
                _propogate_rules(anc,j,k,Models.diff_branches,model_probs)
            elif anc[i+1,k+1] == 0 and anc[k+1,i+1] == 1: # i dec k
                #j cannot be anc k
                model_probs[j,k,Models.A_B]=-np.inf 
                model_probs[k,j,Models.B_A]=-np.inf
            elif anc[i+1,k+1] == 0 and anc[k+1,i+1] == 0: # i brn k
                # j and k can have any relationship
                continue 
            elif anc[j+1,k+1] == 1 and anc[k+1,j+1] == 0: # j anc k
                #i must be brn to k
                _propogate_rules(anc,i,k,Models.diff_branches,model_probs) 
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 1: # j dec k
                #i must not be anc k
                model_probs[i,k,Models.A_B]=-np.inf 
                model_probs[k,i,Models.B_A]=-np.inf 
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 0: # j brn k
                # j and k can have any relationship
                continue 
            
            if np.sum(model_probs[i,k,:]==np.NINF)==4:
                rel = np.argwhere(model_probs[i,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,i,k,rel,model_probs)
            if np.sum(model_probs[j,k,:]==np.NINF)==4:
                rel = np.argwhere(model_probs[j,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,j,k,rel,model_probs)
    return

@numba.njit(cache=True)
def _make_selection(selection_probs, samp_prob):
    #  This is basically a fancy np.random.choice that works well with numba
    #  and unnormalized probabilities stored in a tensor.
    # maxP = np.max(selection_probs)
    norm_sp = np.copy(selection_probs)
    nrmC = 0
    for i in range(norm_sp.shape[0]):
        for j in range(norm_sp.shape[1]):
            max_ij = np.max(norm_sp[i,j,:])
            if max_ij == np.NINF:
                norm_sp[i,j,:] = 0
                continue
            for rel in range(norm_sp.shape[2]):
                norm_sp[i,j,rel] = np.exp(norm_sp[i,j,rel] - max_ij)
                nrmC = nrmC + norm_sp[i,j,rel]

    s = 0
    a = np.random.rand()*nrmC
    choice_made = False
    for i in range(norm_sp.shape[0]):
        for j in range(norm_sp.shape[1]):
            for rel in range(norm_sp.shape[2]):
                if a < s + norm_sp[i,j,rel]:
                    choice_made = True
                    samp_prob += norm_sp[i,j,rel]
                    break
                s = s + norm_sp[i,j,rel]
            if choice_made:
                break
        if choice_made:
                break
    return i,j,rel, samp_prob


def _sample_tree(pairs_tensor):

    rels = {Models.A_B: "anc", Models.B_A: "dec", Models.diff_branches: "branched"}

    n_mut = pairs_tensor.shape[0]
    anc = np.full((n_mut+1,n_mut+1), -1, np.int8)
    anc[0,:] = 1
    anc[:,0] = 0
    np.fill_diagonal(anc,1)
    selection_probs = np.copy(pairs_tensor)
    selection_probs[range(n_mut),range(n_mut),:] = -np.inf
    selection_probs[:,:,Models.cocluster] = -np.inf
    selection_probs[:,:,Models.garbage] = -np.inf
    samp_prob = 0
    while np.any(anc==-1):
        i,j,rel, samp_prob = _make_selection(selection_probs,samp_prob)
        _propogate_rules(anc,i,j,rel,selection_probs)

    return anc, samp_prob


@numba.njit(cache=True)
def _sample_rel(selection_probs, samp_prob):
    norm_sp = np.exp(selection_probs - np.max(selection_probs))
    norm_sp = norm_sp / np.sum(norm_sp)
    p = np.random.rand()
    s = 0
    for rel in range(NUM_MODELS):
        if p < s + norm_sp[rel]:
            samp_prob += np.log(norm_sp[rel])
            break
        s = s + norm_sp[rel]
    return rel, samp_prob


def _sample_tree_w_pair_order(pairs_tensor, order_by_certainty=False):

    n_mut = pairs_tensor.shape[0]
    anc = np.full((n_mut+1,n_mut+1), -1, np.int8)
    anc[0,:] = 1
    anc[:,0] = 0
    np.fill_diagonal(anc,1)
    selection_probs = np.copy(pairs_tensor)
    # for i in range(n_mut):
    #     selection_probs[i,i,:] = -np.inf
    # selection_probs[:,:,Models.cocluster] = -np.inf
    # selection_probs[:,:,Models.garbage] = -np.inf
    if order_by_certainty:
        max_ps = np.max(selection_probs,axis=2)
        rng_i, rng_j = np.unravel_index(np.argsort(-max_ps, axis=None),shape=(n_mut,n_mut))
    else:
        rng_i = np.zeros(n_mut*n_mut,dtype=np.int)
        rng_j = np.zeros(n_mut*n_mut,dtype=np.int)
        cnt = 0
        for i in range(n_mut):
            for j in range(n_mut):
               rng_i[cnt] = i
               rng_j[cnt] = j
               cnt += 1 
    samp_prob = 0
    for i,j in zip(rng_i, rng_j):
        if np.all(selection_probs[i,j,:] == np.NINF):
            continue
        rel, samp_prob = _sample_rel(selection_probs[i,j,:], samp_prob)
        _propogate_rules(anc,i,j,rel,selection_probs)

    return anc, samp_prob


def sample_trees(pairs_tensor, n_samples, order_by_certainty=True, parallel=None):
    #Only use the pairs_tensor which has been normalized ignoring cocluster and garbage models, since we do not want to select them.
    assert np.all(pairs_tensor[:,:,Models.cocluster] == np.NINF)
    assert np.all(pairs_tensor[:,:,Models.garbage] == np.NINF)
    assert np.all(pairs_tensor[range(n_muts),range(n_muts),:] == np.NINF)
    assert np.all(np.isclose(np.sum(np.exp(pairs_tensor),axis=2),1))


    n_muts = pairs_tensor.shape[0]
    trees = np.zeros((n_muts+1, n_muts+1, n_samples))
    tree_probs = np.zeros(n_samples)
    if parallel is None:
        for i in range(n_samples):
            trees[:,:,i], tree_probs[i] = _sample_tree_w_pair_order(pairs_tensor, order_by_certainty=order_by_certainty)
    else:
        pool = multiprocessing.Pool(parallel)
        s = []
        for i in range(n_samples):
            s.append(pool.apply_async(_sample_tree_w_pair_order, args=(pairs_tensor, order_by_certainty)))
        pool.close()
        pool.join()
        for i in range(n_samples):
            trees[:,:,i], tree_probs[i] = s[i].get()

    return trees, tree_probs


def main():
    #FOR DEBUGGING PURPOSES
    import numpy as np
    import sys, os
    sys.path.append(os.path.abspath('../../lib'))
    import pairs_tensor_constructor

    from data_simulator_full_auto import generate_simulated_data

    data, true_tree = generate_simulated_data(n_clust=50, 
                                            n_cells=100, 
                                            n_muts=50, 
                                            FPR=0.001, 
                                            ADO=0.1, 
                                            cell_alpha=1, 
                                            mut_alpha=1,
                                            drange=1
                                            )
    adj_mat = true_tree[1]


    pairs_tensor = pairs_tensor_constructor.construct_pairs_tensor(data,0.001,0.1,1, verbose=False)
    pairs_tensor = np.exp(pairs_tensor)

    sample = _sample_tree(pairs_tensor)

    return


if __name__ == "__main__":
    main()