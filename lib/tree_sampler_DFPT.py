import numpy as np
import numba
import multiprocessing
import time
import queue
import itertools
import tree_util

from common import Models, NUM_MODELS
from util import numba_logsumexp
# from tree_util import calc_tree_llh, convert_ancmatrix_to_parents, convert_parents_to_adjmatrix, convert_adjmatrix_to_ancmatrix, compute_node_relations
from progressbar import progressbar


def calc_posterior_norm_constant(llhs, samp_probs):
    n_samples = len(llhs)
    nlogC = -np.log(n_samples) + numba_logsumexp(llhs - samp_probs)
    return nlogC

@numba.njit
def calc_importance_sampling_val(samples, ps, gs):
    n_samp = samples.shape[0]
    res = 0
    ratios = ps - gs
    ratios = np.exp(ratios - np.max(ratios))
    for s in range(n_samp):
        res = res + samples[s]*ratios[s]
    return res

# @numba.njit(cache=True)
def calc_importance_sampling_matrix(samples, ps, gs):
    n_samp = samples.shape[0]
    res = np.zeros(samples.shape[1:])
    log_ratios = ps - gs
    norm_factor = numba_logsumexp(log_ratios)
    log_ratios = log_ratios - norm_factor
    ratios = np.exp(log_ratios - np.max(log_ratios))
    for s in range(n_samp):
        res = res + samples[s,:]*ratios[s]
    res = res*np.exp(np.max(log_ratios))
    return res

@numba.njit
def calc_importance_sampling_pairs_tensor(samples, ps, gs):
    n_samp = samples.shape[0]
    n_mut = samples.shape[1]
    res = np.zeros((n_mut, n_mut, NUM_MODELS))
    ratios = ps - gs
    ratios = np.exp(ratios - np.max(ratios))
    for s in range(n_samp):
        sample = samples[s,:]
        adj = tree_util.convert_parents_to_adjmatrix(sample)
        node_rels = tree_util.compute_node_relations(adj)
        for i in range(n_mut):
            for j in range(n_mut):
                res[i,j,node_rels[i+1,j+1]] += ratios[s]
    for i in range(n_mut):
        for j in range(n_mut):
            res[i,j,:] /= np.sum(res[i,j,:])
    return res


def calc_some_importance_sampling_values(samples, samp_probs, data, fprs, ados, mut_ass, verbose=True):
    n_samples = samples.shape[0]
    n_mut = samples.shape[1]
    llhs = np.zeros(n_samples)
    for i in range(n_samples):
        if verbose and i % int(n_samples/20) == 0:
            print(i, "/", n_samples)
        adj = tree_util.convert_parents_to_adjmatrix(samples[i,:])
        anc = tree_util.convert_adjmatrix_to_ancmatrix(adj)
        llhs[i] = tree_util.calc_tree_llh(data, anc, mut_ass, fprs, ados, 1)
        
    ratios = llhs - samp_probs
    ratios = np.exp(ratios - np.max(ratios))

    IS_anc_mat = np.zeros((n_mut+1, n_mut+1))
    IS_adj_mat = np.zeros((n_mut+1, n_mut+1))
    for i in range(n_samples):
        adj = tree_util.convert_parents_to_adjmatrix(samples[i,:])
        anc = tree_util.convert_adjmatrix_to_ancmatrix(adj)
        
        IS_adj_mat = IS_adj_mat + adj*ratios[i]
        IS_anc_mat = IS_anc_mat + anc*ratios[i]
    
    IS_adj_mat = IS_adj_mat/n_samples
    IS_anc_mat = IS_anc_mat/n_samples

    return llhs, IS_adj_mat, IS_anc_mat


def calc_IS_anc_mat(samples, log_posts, log_qs):
    n_samples, n_mut = samples.shape
    IS_anc_mat = np.zeros((n_mut+1,n_mut+1))
    for i in range(n_samples):
        IS_anc_mat += np.exp(log_posts[i] - log_qs[i]) * tree_util.convert_parents_to_ancmatrix(samples[i,:])
    IS_anc_mat = IS_anc_mat / n_samples
    return IS_anc_mat

def calc_IS_adj_mat(samples, log_posts, log_qs):
    n_samples, n_mut = samples.shape
    IS_adj_mat = np.zeros((n_mut+1,n_mut+1))
    for i in range(n_samples):
        IS_adj_mat += np.exp(log_posts[i] - log_qs[i]) * tree_util.convert_parents_to_adjmatrix(samples[i,:])
    IS_adj_mat = IS_adj_mat / n_samples
    return IS_adj_mat


def calc_sample_posts(samples, samp_probs, data, fprs, ados, mut_ass, d_rng_i):
    n_samples = len(samples)

    llhs = np.zeros(n_samples)
    for i in range(n_samples):
        uniq_anc = tree_util.convert_parents_to_ancmatrix(samples[i])
        llhs[i] = tree_util.calc_tree_llh(data,uniq_anc,mut_ass,fprs,ados,d_rng_i)
    
    logC = np.log(n_samples) - numba_logsumexp(llhs - samp_probs)
    log_posts = logC + llhs

    return log_posts


@numba.njit(cache=True)
def _propogate_rules(anc,i,j,rel,model_probs):
    # If set i branched to j, then:
    #  - All nodes desended from i are branched to all nodes desended from j
    #  - All nodes dec(i) cannot be anc to all nodes anc(j)
    #  - All nodes dec(j) cannot be anc to all nodes anc(i)
    # If set i ancestral to j, then:
    #  - All nodes anc(i) will be ancestral to all nodes dec(j)
    #  - All nodes branch(i) will be branched to all nodes dec(j)
    #  - All nodes anc(i) cannot be branched to all nodes anc(j)
    # If set i desendant to j, then equiv to set j ancestral i

    if rel == Models.A_B:
        a,b = 1,0
    elif rel == Models.B_A:
        a,b = 0,1
    elif rel == Models.diff_branches:
        a,b = 0,0

    anc[i+1,j+1] = a
    anc[j+1,i+1] = b
    model_probs[i,j,:] = -np.inf
    model_probs[j,i,:] = -np.inf
    n_muts = model_probs.shape[0]


    if rel == Models.A_B or rel == Models.B_A:
        if rel == Models.B_A:
            i,j = j,i
        for k in range(n_muts):
            if k == i or k == j:
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
            
            #If after propogation there is only one valid relationship, set that relationship and propogate againt
            if np.sum(model_probs[i,k,:]==np.NINF)==4:
                rel = np.argwhere(model_probs[i,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,i,k,rel,model_probs)
            if np.sum(model_probs[j,k,:]==np.NINF)==4:
                rel = np.argwhere(model_probs[j,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,j,k,rel,model_probs)
    elif rel == Models.diff_branches:
        for k in range(n_muts):
            if k == i or k == j:
                continue
            if not np.logical_xor(anc[i+1,k+1] == -1, anc[j+1,k+1] == -1):
                continue

            if anc[i+1,k+1] == 0 and anc[k+1,i+1] == 0: # i brn k
                # j and k can have any relationship
                continue 
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 0: # j brn k
                # j and k can have any relationship
                continue 
            elif anc[i+1,k+1] == 1 and anc[k+1,i+1] == 0: # i anc k
                #j must be brn from k
                _propogate_rules(anc,j,k,Models.diff_branches,model_probs)
            elif anc[j+1,k+1] == 1 and anc[k+1,j+1] == 0: # j anc k
                #i must be brn to k
                _propogate_rules(anc,i,k,Models.diff_branches,model_probs) 
            elif anc[i+1,k+1] == 0 and anc[k+1,i+1] == 1: # i dec k
                #j cannot be anc k
                model_probs[j,k,Models.A_B]=-np.inf 
                model_probs[k,j,Models.B_A]=-np.inf
            elif anc[j+1,k+1] == 0 and anc[k+1,j+1] == 1: # j dec k
                #i must not be anc k
                model_probs[i,k,Models.A_B]=-np.inf 
                model_probs[k,i,Models.B_A]=-np.inf
            
            if np.sum(model_probs[i,k,:]==np.NINF)==4:
                rel = np.argwhere(model_probs[i,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,i,k,rel,model_probs)
            if np.sum(model_probs[j,k,:]==np.NINF)==4:
                rel = np.argwhere(model_probs[j,k,:].flatten() > -np.inf).flatten()[0]
                _propogate_rules(anc,j,k,rel,model_probs)
    return


@numba.njit(cache=True)
def _sample_rel(selection_probs):
    #With numba, this is faster than calling the commented out line below
    max_sp = -np.inf
    for rel in range(NUM_MODELS):
        if selection_probs[rel] > max_sp:
            max_sp = selection_probs[rel]
    norm_sp = np.exp(selection_probs - max_sp)
    # norm_sp = np.exp(selection_probs - np.max(selection_probs))

    #With numba, this is faster than calling the commented out line below
    sum_nsp = 0
    for rel in range(NUM_MODELS):
        sum_nsp += norm_sp[rel]
    norm_sp = norm_sp / sum_nsp
    # norm_sp = norm_sp / np.sum(norm_sp)

    p = np.random.rand()
    s = 0
    sel_prob = 0
    for rel in range(NUM_MODELS):
        this_rel_p = norm_sp[rel]
        if p < s + this_rel_p:
            sel_prob += np.log(this_rel_p)
            break
        s = s + this_rel_p

    return rel, sel_prob


numba.njit(cache=True)
def _sample_tree(pairs_tensor, rng_i, rng_j, status=None):
    n_mut = pairs_tensor.shape[0]
    anc = np.full((n_mut+1,n_mut+1), -1, np.int8)
    anc[0,:] = 1
    anc[:,0] = 0
    for i in range(n_mut+1):
        anc[i,i] = 1
    selection_probs = np.copy(pairs_tensor)
    samp_prob = 0
    for i,j in zip(rng_i, rng_j):
        if np.all(selection_probs[i,j,:] == np.NINF):
            continue
        rel, sel_prob = _sample_rel(selection_probs[i,j,:])
        samp_prob += sel_prob
        _propogate_rules(anc,i,j,rel,selection_probs)

    if status is not None:
        status.put(True)

    return tree_util.convert_ancmatrix_to_parents(anc), samp_prob

numba.njit(cache=True)
def _sample_trees(pairs_tensor, rng_i, rng_j, n_samples, status=None):
    n_mut = pairs_tensor.shape[0]
    trees = np.zeros((n_samples, n_mut), dtype=int)
    tree_probs = np.zeros(n_samples)
    for i in range(n_samples):
        tree, tree_prob = _sample_tree(pairs_tensor, rng_i, rng_j, status)
        trees[i,:] = tree
        tree_probs[i] = tree_prob
    return trees, tree_probs

def _calc_sample_llhs(samples, data, mut_ass, fprs, ados, d_rng_i):
    n_samples = samples.shape[0]
    llhs = np.zeros(n_samples)
    for i in range(n_samples):
        anc = tree_util.convert_parents_to_ancmatrix(samples[i])
        llhs[i] = tree_util.calc_tree_llh(data,anc,mut_ass,fprs,ados,d_rng_i)
    return llhs

def calc_sample_llhs(samples, data, mut_ass, fprs, ados, d_rng_i, parallel=None):
    n_samples = samples.shape[0]
    llhs = np.zeros(n_samples)
    if parallel is None:
        llhs = _calc_sample_llhs(samples, data, mut_ass, fprs, ados, d_rng_i)
    else:
        with multiprocessing.Pool(parallel) as pool:
            jobs = []
            chunksize = int(np.max([1,np.min([100, np.ceil(n_samples/parallel)])]))
            for i in range(int(np.floor(n_samples/chunksize))):
                jobs.append(pool.apply_async(_calc_sample_llhs, args=(samples[chunksize*i:chunksize*(i+1),:], data, mut_ass, fprs, ados, d_rng_i)))
            
            jobs.append(pool.apply_async(_calc_sample_llhs, args=(samples[chunksize*(i+1):,:], data, mut_ass, fprs, ados, d_rng_i)))
            pool.close()
            pool.join()
            for i in range(int(np.floor(n_samples/chunksize))):
                llhs[chunksize*i:chunksize*(i+1)] = jobs[i].get()
            llhs[chunksize*(i+1):] = jobs[i+1].get()

    return llhs


def sample_trees(pairs_tensor, n_samples, order_by_certainty=True, parallel=None):
    #Only use the pairs_tensor which has been normalized ignoring cocluster and garbage models, since we do not want to select them.
    n_mut = pairs_tensor.shape[0]
    # assert np.all(pairs_tensor[:,:,Models.cocluster] == np.NINF)
    assert np.all(pairs_tensor[:,:,Models.garbage] == np.NINF)
    assert np.all(np.isclose(np.sum(np.exp(pairs_tensor),axis=2),1))

    if order_by_certainty:
        #If we order the mutation pairs by our certainty in any one of the possible relationships
        #then we sample these mutations first and so there is little chance of these high-probability
        #relationships being forceably removed through rule propogation during selection of lower
        #certainly mutation pairs. I.e., this will result in more high-confidence trees being sampled.
        max_ps = np.max(pairs_tensor,axis=2)
        rng_i, rng_j = np.unravel_index(np.argsort(-max_ps, axis=None),shape=(n_mut,n_mut))
    else:
        #However, we may be worried about overconstraining the tree search space, so we can use an 
        #arbitrary mutation pair ordering set here.
        rngs = np.array([[i,j] for i in range(n_mut) for j in range(n_mut)])
        rng_i, rng_j = rngs[:,0], rngs[:,1]
    

    selection_probs = np.copy(pairs_tensor)
    selection_probs[range(n_mut),range(n_mut),Models.cocluster] = -np.inf

    trees = np.zeros((n_samples, n_mut), dtype=int)
    tree_probs = np.zeros(n_samples)
    
    if parallel is None:
        with progressbar(total=n_samples, desc='Sampling trees', unit='tree', dynamic_ncols=True) as pbar:
            for i in range(n_samples):
                trees[i,:], tree_probs[i] = _sample_tree(selection_probs, rng_i, rng_j)
                pbar.update()
    else:
        manager = multiprocessing.Manager()
        sample_status_queue = manager.Queue()
        pool = multiprocessing.Pool(parallel)
        
        jobs = []
        chunksize = int(np.max([1,np.min([100, np.ceil(n_samples/parallel)])]))
        for i in range(int(np.floor(n_samples/chunksize))):
            jobs.append(pool.apply_async(_sample_trees, args=(selection_probs, rng_i, rng_j, chunksize, sample_status_queue)))
        remainder = n_samples % chunksize
        jobs.append(pool.apply_async(_sample_trees, args=(selection_probs, rng_i, rng_j, remainder, sample_status_queue)))
        pool.close()

        with progressbar(total=n_samples, desc='Sampling trees', unit='tree', dynamic_ncols=True) as pbar:
            n_sampled = 0
            while n_sampled < n_samples:
                
                if np.all([i.ready() for i in jobs]):
                    break
                try:
                    # If there's something in the queue for us to retrieve, a child
                    # process has sampled a tree.
                    sample_status_queue.get(timeout=1)
                    n_sampled += 1
                    pbar.update()

                except queue.Empty:
                    time.sleep(0.5)
                
        
        pool.join()
        for i in range(int(np.floor(n_samples/chunksize))):
            trees[chunksize*i:chunksize*(i+1),:], tree_probs[chunksize*i:chunksize*(i+1)] = jobs[i].get()
        trees[chunksize*(i+1):,:], tree_probs[chunksize*(i+1):] = jobs[i+1].get()
        pool.terminate()
    return trees, tree_probs


def main():
    print("tree_sampler_DFPT is not callable by itself. Either use call bin/sc_pairtree or import sample_trees() into another script.")
    pass


if __name__ == "__main__":
    main()