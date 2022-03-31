import numpy as np
from numba import njit
import math
from scipy.special import logsumexp
import time

from progressbar import progressbar
import hyperparams as hparams
import util
import common
from common import Models, DataRange, DataRangeIdx
from pairs_tensor_util import p_data_given_truth_and_errors

from collections import namedtuple
TreeSample = namedtuple('TreeSample', (
  'adj',
  'anc',
  'llh'
))


# @njit
def _calc_tree_llh(data, anc, FPR, ADO, dtype=DataRangeIdx.ref_var_nodata):
    #First, I need to calculate the number of true positives/negatives, and false positives/negatives for
    #the case of each cell being assigned to each node.
    #Note: matrix multiplication is optimized when the elements are floats, but not if they are integers.
    # --> Swap to using floats for these calcs.
    
    d_set = DataRange[dtype]
    n_mut, n_cell = data.shape
    assert len(FPR) == n_mut
    assert len(ADO)  == n_mut
    
    cell_at_mut_contribution = np.zeros((n_mut,n_cell))
    for t in [0,1]:
        anc_comp = (anc[1:,1:]==t) + 0.0
        for d in d_set:
            D_comp = (data == d) + 0.0
            
            p_dgte = np.array([p_data_given_truth_and_errors(d,t,FPR[i],ADO[i],dtype) for i in range(n_mut)])[np.newaxis].T @ np.ones((1,n_cell)) #Probability of datapoint given hidden truth and error rates P(d|t,Theta)
            n_dgt  = anc_comp.T @ D_comp #Determines for each cell, the number of datapoints within it called d with given truth t, with t determined by a cell being assigned to each node in the given tree.
            cell_at_mut_contribution += n_dgt*np.log(p_dgte)
    tree_llh = np.sum(logsumexp(cell_at_mut_contribution,axis=0))
    return tree_llh


@njit
def make_ancestral_from_adj(adj, check_validity=False):
    #NOTE: This code was copied from Jeff's util function.
    # --> Move to it's own util function!!

    K = len(adj)
    root = 0

    if check_validity:
        # By default, disable checks to improve performance.
        assert np.all(1 == np.diag(adj))
        expected_sum = 2 * np.ones(K)
        expected_sum[root] = 1
        assert np.array_equal(expected_sum, np.sum(adj, axis=0))

    Z = np.copy(adj)
    np.fill_diagonal(Z, 0)
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
    np.fill_diagonal(Z, 1)

    if check_validity:
        assert np.array_equal(Z[root], np.ones(K))
    return Z


@njit
def _calc_tree_logmutrel(adj, pairs_tensor):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    node_rels = util.compute_node_relations(adj)
    K = len(node_rels)
    assert node_rels.shape == (K, K)
    assert pairs_tensor.shape == (K-1, K-1, 5) #JB Note: I changed last input to 5... assuming that 5 models will be used

    # First row and column of `tree_logmutrel` will always be zero.
    tree_logmutrel = np.zeros((K,K))
    rng = range(K-1)
    for J in rng:
        for K in rng:
            JK_clustrel = node_rels[J+1,K+1]
            tree_logmutrel[J+1,K+1] = pairs_tensor[J,K,JK_clustrel]

    assert np.array_equal(tree_logmutrel, tree_logmutrel.T)
    assert np.all(tree_logmutrel <= 0)
    return tree_logmutrel

def _scaled_softmax(A, R=100):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    #Also, I feel like this should be in util or something

    # Ensures `max(softmax(A)) / min(softmax(A)) <= R`.
    #
    # Typically, I use this as a "softer softmax", ensuring that the largest
    # element in the softmax is at most 100x the magnitude of the smallest.
    # Otherwise, given large differences between the minimum and maximum values,
    # the softmax becomes even more sharply peaked, with one element absorbing
    # effectively all mass.
    noninf = np.logical_not(np.isinf(A))
    if np.sum(noninf) == 0:
        return util.softmax(A)
    delta = np.max(A[noninf]) - np.min(A[noninf])
    if np.isclose(0, delta):
        return util.softmax(A)
    B = min(1, np.log(R) / delta)
    return util.softmax(B*A)

def _sample_cat(W):
    #NOTE: taken from Jeff's tree_sampler. (could probably go into a util file)
    assert np.all(W >= 0) and np.isclose(1, np.sum(W))
    choice = np.random.choice(len(W), p=W)
    assert W[choice] > 0
    return choice

def _init_cluster_adj_mutrels(pairs_tensor):
    K = pairs_tensor.shape[0] + 1
    adj = np.eye(K, dtype=np.int)
    in_tree = set((0,))
    remaining = set(range(1, K))

    W_nodes = np.zeros(K)

    while len(remaining) > 0:
        nodeidxs = np.array(sorted(remaining))
        relidxs = nodeidxs - 1
        assert np.all(relidxs >= 0)
        anc_logprobs = pairs_tensor[np.ix_(relidxs, relidxs)][:,:,Models.A_B]
        # These values are currently -inf, but we must replace these so that joint
        # probabilities of being ancestral to remaining don't all become 0.
        np.fill_diagonal(anc_logprobs, 0)

        assert anc_logprobs.shape == (len(remaining), len(remaining))
        log_W_nodes_remaining = np.sum(anc_logprobs, axis=1)
        # Use really "soft" softmax.
        W_nodes[nodeidxs] = _scaled_softmax(log_W_nodes_remaining)

        # Root should never be selected.
        assert W_nodes[0] == 0
        assert np.isclose(1, np.sum(W_nodes))
        assert np.all(W_nodes[list(in_tree)] == 0)
        nidx = _sample_cat(W_nodes)

        log_W_parents = np.full(K, -np.inf)
        others = np.array(sorted(remaining - set((nidx,))))
        truncated_pairs_tensor = util.remove_rowcol(pairs_tensor,others-1)
    
        for parent in in_tree:
            new_adj = np.copy(adj)
            new_adj[parent,nidx] = 1
            truncated_adj = util.remove_rowcol(new_adj, others)
            tree_logmutrel = _calc_tree_logmutrel(truncated_adj, truncated_pairs_tensor)
            log_W_parents[parent] = np.sum(np.triu(tree_logmutrel))
        W_parents = _scaled_softmax(log_W_parents)
        assert np.all(W_parents[nodeidxs] == 0)
        pidx = _sample_cat(W_parents)
        adj[pidx,nidx] = 1

        remaining.remove(nidx)
        in_tree.add(nidx)
        W_nodes[nidx] = 0

    assert np.all(W_nodes == 0)
    assert len(in_tree) == K
    assert len(remaining) == 0
    return adj

def _init_cluster_adj_branching(K):
    #NOTE: This code was copied from Jeff's tree_sampler.py

    cluster_adj = np.eye(K, dtype=np.int)
    # Every node comes off node 0, which will always be the tree root. Note that
    # we don't assume that the first cluster (node 1, cluster 0) is the clonal
    # cluster -- it's not treated differently from any other nodes/clusters.
    cluster_adj[0,:] = 1
    return cluster_adj

def _make_W_nodes_mutrel(adj, anc, pairs_tensor):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    K = len(adj)
    assert adj.shape == (K, K)

    tree_logmutrel = _calc_tree_logmutrel(adj, pairs_tensor)
    pair_error = 1 - np.exp(tree_logmutrel)
    #pair_error *= 1 - anc

    assert np.allclose(0, np.diag(pair_error))
    assert np.allclose(0, pair_error[0])
    assert np.allclose(0, pair_error[:,0])
    pair_error = np.maximum(common._EPSILON, pair_error)
    node_error = np.sum(np.log(pair_error), axis=1)
    #Jarry edit out... maybe I'll put that debuging stuff back in later.
    # if common.debug.DEBUG:
    #     _make_W_nodes_mutrel.node_error = node_error

    weights = np.zeros(K)
    weights[1:] += _scaled_softmax(node_error[1:])
    weights[1:] += common._EPSILON
    weights /= np.sum(weights)
    assert weights[0] == 0

    return weights

def _make_W_nodes_uniform(adj, anc):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    K = len(adj)
    weights = np.ones(K)
    weights[0] = 0
    weights /= np.sum(weights)
    return weights


def _make_W_dests_mutrel(subtree_head, curr_parent, adj, anc, pairs_tensor):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    assert subtree_head > 0
    assert adj[curr_parent,subtree_head] == 1
    cluster_idx = subtree_head - 1
    K = len(adj)

    logweights = np.full(K, -np.inf)
    for dest in range(K):
        if dest == curr_parent:
            continue
        if dest == subtree_head:
            continue
        new_adj = _modify_tree(adj, anc, dest, subtree_head)
        tree_logmutrel = _calc_tree_logmutrel(new_adj, pairs_tensor)
        logweights[dest] = np.sum(np.triu(tree_logmutrel))
    assert not np.any(np.isnan(logweights))
    valid_logweights = np.delete(logweights, (curr_parent, subtree_head))
    assert not np.any(np.isinf(valid_logweights))

    weights = _scaled_softmax(logweights)
    # Since we end up taking logs, this can't be exactly zero. If the logweight
    # is extremely negative, then this would otherwise be exactly zero.
    weights += common._EPSILON
    weights[curr_parent] = 0
    weights[subtree_head] = 0
    weights /= np.sum(weights)
    return weights

def _make_W_dests_uniform(subtree_head, curr_parent, adj, anc):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    K = len(adj)
    weights = np.ones(K)
    weights[subtree_head] = 0
    weights[curr_parent] = 0
    weights /= np.sum(weights)
    return weights


def _make_W_nodes_combined(adj, anc, pairs_tensor):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    W_nodes_uniform = _make_W_nodes_uniform(adj, anc)
    W_nodes_mutrel = _make_W_nodes_mutrel(adj, anc, pairs_tensor)
    return np.vstack((W_nodes_uniform, W_nodes_mutrel))


def _make_W_dests_combined(subtree_head, adj, anc, pairs_tensor):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    curr_parent = _find_parent(subtree_head, adj)
    W_dests_uniform = _make_W_dests_uniform(subtree_head, curr_parent, adj, anc)
    # s = time.time()
    W_dests_mutrel = _make_W_dests_mutrel(subtree_head, curr_parent, adj, anc, pairs_tensor)
    # e = time.time()
    # print("Time to select dest (mutrel):", e-s)
    return np.vstack((W_dests_uniform, W_dests_mutrel))


def _find_parent(node, adj):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    col = np.copy(adj[:,node])
    col[node] = 0
    parents = np.flatnonzero(col)
    assert len(parents) == 1
    return parents[0]

@njit
def _modify_tree(adj, anc, A, B):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    '''If `B` is ancestral to `A`, swap nodes `A` and `B`. Otherwise, move
    subtree `B` under `A`.

    `B` can't be 0 (i.e., the root node), as we want always to preserve the
    property that node zero is root.'''
    K = len(adj)
    # Ensure `B` is not zero.
    assert 0 <= A < K and 0 < B < K
    assert A != B

    adj = np.copy(adj)

    assert np.array_equal(np.diag(adj), np.ones(K))
    # Diagonal should be 1, and every node except one of them should have a parent.
    assert np.sum(adj) == K + (K - 1)
    # Every column should have two 1s in it corresponding to self & parent,
    # except for column denoting root.
    #NOTE: JB commented out the next assert cause that was holding back @njit,
    #      and after applying njit everything sped up by 10x!
    # assert np.array_equal(np.sort(np.sum(adj, axis=0)), np.array([1] + (K - 1)*[2]))

    np.fill_diagonal(adj, 0)

    if anc[B,A]:
        adj_BA = adj[B,A]
        assert anc[A,B] == adj[A,B] == 0
        if adj_BA:
            adj[B,A] = 0

        # Swap position in tree of A and B. I need to modify both the A and B
        # columns.
        acol, bcol = np.copy(adj[:,A]), np.copy(adj[:,B])
        arow, brow = np.copy(adj[A,:]), np.copy(adj[B,:])
        adj[A,:], adj[B,:] = brow, arow
        adj[:,A], adj[:,B] = bcol, acol

        if adj_BA:
            adj[A,B] = 1
            #debug('tree_permute', (A,B), 'swapping', A, B)
    else:
        # Move B so it becomes child of A. I don't need to modify the A column.
        adj[:,B] = 0
        adj[A,B] = 1
        #debug('tree_permute', (A,B), 'moving', B, 'under', A)

    np.fill_diagonal(adj, 1)
    return adj


def _generate_new_sample(old_samp, data, pairs_tensor, FPR, ADO, d_rng_id):
    #NOTE: This code was copied from Jeff's tree_sampler.py
    #I removed last two inputs for calculating the phis.

    K = len(old_samp.adj)
    # When a tree consists of two nodes (i.e., one mutation cluster), proceeding with
    # the normal sample-generating process will produce an error (specifically,
    # when we try to divide by zero in _make_W_dests_uniform). Circumvent this by
    # returning the current (trivial) tree structure.
    if K == 2:
        return (old_samp, 0., 0.)

    # mode == 0: make uniform update
    # mode == 1: make mutrel-informed update
    mode_node_weights = np.array([1 - hparams.gamma, hparams.gamma])
    mode_dest_weights = np.array([1 - hparams.zeta,  hparams.zeta])
    mode_node = _sample_cat(mode_node_weights)
    mode_dest = _sample_cat(mode_dest_weights)

    #Calc the weights of choosing the node to move using the old tree.
    s = time.time()
    W_nodes_old = _make_W_nodes_combined(old_samp.adj, old_samp.anc, pairs_tensor)
    e = time.time()
    # print("Getting node weights for one to move:", e-s)
    #Choose a node to move.
    B = _sample_cat(W_nodes_old[mode_node])
    #Calc the weights of choosing the destination for the node to move to.
    s = time.time()
    W_dests_old = _make_W_dests_combined(
        B,
        old_samp.adj,
        old_samp.anc,
        pairs_tensor,
    )
    e = time.time()
    # print("Getting node weights for where to move:", e-s)

    #Choose a destination for the node to move to.
    A = _sample_cat(W_dests_old[mode_dest])
    #A = _find_parent(B, common._true_adjm)
    #Make the move and update the tree llh
    s = time.time()
    new_adj = _modify_tree(old_samp.adj, old_samp.anc, A, B)
    e = time.time()
    # print("Modifying tree once:", e-s)

    new_anc = make_ancestral_from_adj(new_adj)
    s = time.time()
    new_samp = TreeSample(
        adj = new_adj,
        anc = new_anc,
        llh = _calc_tree_llh(data, new_anc, FPR, ADO, d_rng_id)
    )
    e = time.time()
    # print("Calcing tree llh:", e-s)


    # `A_prime` and `B_prime` correspond to the node choices needed to reverse
    # the tree perturbation.
    if old_samp.anc[B,A]:
        # If `B` is ancestral to `A`, the tree perturbation swaps the nodes. Thus,
        # simply invert the swap to reverse the move.
        A_prime = B
        B_prime = A
    else:
        # If `B` isn't ancestral to `A`, the tree perturbation moves the subtree
        # headed by `B` so that `A` becomes its parent. To reverse the move, move
        # the `B` subtree back under its old parent.
        A_prime = _find_parent(B, old_samp.adj)
        B_prime = B
    #Under the new tree, need to recalculate the weights of moving nodes so can calc p of moving back to old tree.
    W_nodes_new = _make_W_nodes_combined(new_samp.adj, new_samp.anc, pairs_tensor)
    W_dests_new = _make_W_dests_combined(
        B_prime,
        new_samp.adj,
        new_samp.anc,
        pairs_tensor,
    )

    # JB: block out (for now)
    # if common.debug.DEBUG:
    #     true_parent = _find_parent(B, common._true_adjm)
    #     old_parent = _find_parent(B, old_samp.adj)
    #     _generate_new_sample.debug = (
    #     (
    #         B,
    #         (A, true_parent, old_parent),
    #     ),
    #     (mode_node, mode_dest),
    #     W_nodes_old[0],
    #     W_nodes_old[1],
    #     W_dests_old[0],
    #     W_dests_old[1],
    #     '%.3f' % np.max(W_nodes_old[1]),
    #     '%.3f' % np.max(W_dests_old[1]),
    #     )
    log_p_B_new_given_old = np.log(np.dot(mode_node_weights, W_nodes_old[:,B]))
    log_p_A_new_given_old = np.log(np.dot(mode_dest_weights, W_dests_old[:,A]))
    # The need to use `A_prime` and `B_prime` here rather than `A` and `B`
    # becomes apparent when you consider the case when `B` is ancestral to `A` in
    # the old tree.
    log_p_B_old_given_new = np.log(np.dot(mode_node_weights, W_nodes_new[:,B_prime]))
    log_p_A_old_given_new = np.log(np.dot(mode_dest_weights, W_dests_new[:,A_prime]))

    log_p_new_given_old = log_p_B_new_given_old + log_p_A_new_given_old
    log_p_old_given_new = log_p_B_old_given_new + log_p_A_old_given_new
    return (new_samp, log_p_new_given_old, log_p_old_given_new)


def _init_chain(seed, data, pairs_tensor, FPR, ADO, d_rng_id):
    #NOTE: This code was copied from Jeff's tree_sampler.py

    # Ensure each chain gets a new random state. I add chain index to initial
    # random seed to seed a new chain, so I must ensure that the seed is still in
    # the valid range [0, 2**32).
    np.random.seed(seed % 2**32)

    ##JB: Let's ignore this part for now and just choose the fully branched tree
    if np.random.uniform() < hparams.iota:
        init_adj = _init_cluster_adj_mutrels(pairs_tensor)
    else:
        # Particularly since clusters may not be ordered by mean VAF, a branching
        # tree in which every node comes off the root is the least biased
        # initialization, as it doesn't require any steps that "undo" bad choices, as
        # in the linear or random (which is partly linear, given that later clusters
        # aren't allowed to be parents of earlier ones) cases.
        K = pairs_tensor.shape[0] + 1
        init_adj = _init_cluster_adj_branching(K)
    
    common.ensure_valid_tree(init_adj)

    init_anc = make_ancestral_from_adj(init_adj)

    init_llh = _calc_tree_llh(data, init_anc, FPR, ADO, d_rng_id)

    init_samp = TreeSample(
        adj = init_adj,
        anc = init_anc,
        llh = init_llh
    )
    return init_samp



def _run_chain(data, pairs_tensor, FPR, ADO, nsamples, thinned_frac, seed, d_rng_id, progress_queue=None):
    #Note: Taken from Jeff's tree_sampler.
    #I am going to strip out a bunch of stuff I don't need (yet).

    assert nsamples > 0

    samps = [_init_chain(seed, data, pairs_tensor, FPR, ADO, d_rng_id)]
    accepted = 0
    if progress_queue is not None:
        progress_queue.put(0)

    assert 0 < thinned_frac <= 1
    record_every = round(1 / thinned_frac)
    # Why is `expected_total_trees` equal to this?
    #
    # We always taken the first tree, since `0%k = 0` for all `k`. There remain
    # `nsamples - 1` samples to take, of which we record every `record_every`
    # one.
    #
    # This can give somewhat weird results, since you intuitively expect
    # approximately `thinned_frac * nsamples` trees to be returned. E.g., if
    # `nsamples = 3000` and `thinned_frac = 0.3`, you expect `0.3 * 3000 = 900`
    # trees, but you actually get 1000. To not be surprised by this, try to
    # choose `thinned_frac` such that `1 / thinned_frac` is close to an integer.
    # (I.e., `thinned_frac = 0.5` or `thinned_frac = 0.3333333` generally give
    # results as you'd expect.
    expected_total_trees = 1 + math.floor((nsamples - 1) / record_every)

    old_samp = samps[0]
    for I in range(1, nsamples):
        if progress_queue is not None:
            progress_queue.put(I)
        new_samp, log_p_new_given_old, log_p_old_given_new = _generate_new_sample(
            old_samp,
            data,
            pairs_tensor,
            FPR,
            ADO,
            d_rng_id
        )
        log_p_transition = (new_samp.llh - old_samp.llh) + (log_p_old_given_new - log_p_new_given_old)
        U = np.random.uniform()
        accept = log_p_transition >= np.log(U)
        if accept:
            samp = new_samp
        else:
            samp = old_samp

        #NOTE: JB removed a print_debug section.

        if I % record_every == 0:
            samps.append(samp)
            old_samp = samp
        if accept:
            accepted += 1

    if nsamples > 1:
        accept_rate = accepted / (nsamples - 1)
    else:
        accept_rate = 1.
    assert len(samps) == expected_total_trees
    return (
        [S.adj   for S in samps],
        [S.llh   for S in samps],
        accept_rate,
    )


def sample_trees(sc_data, pairs_tensor, FPR, ADO, trees_per_chain, burnin, nchains, thinned_frac, seed, parallel, d_rng_id):

    assert nchains > 0
    assert trees_per_chain > 0
    assert 0 <= burnin <= 1
    assert 0 < thinned_frac <= 1

    #debug stuff commented out by Jarry
#   if common.debug.DEBUG:
#       _load_truth(common.debug._truthfn)

    jobs = []
    total = nchains * trees_per_chain

  # Don't use (hard-to-debug) parallelism machinery unless necessary.
    if parallel > 0:
        import concurrent.futures
        import multiprocessing
        import queue
        import time
        import sys

        manager = multiprocessing.Manager()
        # What is stored in progress_queue doesn't matter. The queue is just used
        # so that child processes can signal when they've sampled a tree, allowing
        # the main process to update the progress bar.
        progress_queue = manager.Queue()

        with progressbar(total=total, desc='Sampling trees', unit='tree', dynamic_ncols=True) as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as ex:
                for C in range(nchains):
                    # Ensure each chain's random seed is different from the seed used to
                    # seed the initial Pairtree invocation, yet nonetheless reproducible.
                    jobs.append(ex.submit(_run_chain, sc_data, pairs_tensor, FPR, ADO, trees_per_chain, thinned_frac, seed + C + 1, d_rng_id, progress_queue))

                while True:
                    finished = 0
                    last_check = time.perf_counter()

                    for J in jobs:
                        if J.done():
                            exception = J.exception(timeout=0.001)
                            if exception is not None:
                                # Ideally, if an exception occurs in a child process, we'd like
                                # to re-raise the exception from the parent process and cause
                                # the application to crash immediately. However,
                                # `concurrent.futures` will wait until all child processes
                                # terminate (either because they finished successfully or
                                # suffered an exception) to raise the exception from the parent
                                # process. If an exception occurs in the child process, it
                                # would normally be raised automatically when we call
                                # `J.result()` below; however, since we're in a loop, we must
                                # break out of the loop, which we can do so by explicitly
                                # re-raising the exception.
                                #
                                # Note that the `print()` call will happen immediately, so the
                                # user will be notified as soon as the error occurs. Actually
                                # raising the exception and crashing the application will not
                                # occur, however, until all child processes finish. When the
                                # `raise` happens below, the parent process will effectively
                                # freeze at that statement until all child processes finish,
                                # such that the progress bar stops updating. Ieally, we could
                                # terminate all child processes when we detect the exception,
                                # but we would have to do this manually by sending SIGTERM to
                                # the processes. It's evidently impossible to get the PIDs of
                                # the child processes without installing `psutil`, so we will
                                # just allow the application to wait until child processes
                                # finish before crashing.
                                print('Exception occurred in child process:', exception, file=sys.stderr)
                                raise exception
                            else:
                                finished += 1

                    if finished == nchains:
                        break
                    while time.perf_counter() - last_check < 1:
                        try:
                            # If there's something in the queue for us to retrieve, a child
                            # process has sampled a tree.
                            progress_queue.get(timeout=1)
                            pbar.update()
                        except queue.Empty:
                            pass

        results = [J.result() for J in jobs]
    else:
        results = []
        for C in range(nchains):
            results.append(_run_chain(sc_data, pairs_tensor, FPR, ADO, trees_per_chain, thinned_frac, seed + C + 1, d_rng_id))

    merged_adj = []
    merged_llh = []
    accept_rates = []
    for A, L, accept_rate in results:
        assert len(A) == len(L) == len(results[0][0])
        discard_first = round(burnin * len(A))
        merged_adj += A[discard_first:]
        merged_llh += L[discard_first:]
        accept_rates.append(accept_rate)
    assert len(merged_adj) == len(merged_llh)
    return (merged_adj, merged_llh, accept_rates)


def compute_posterior(adjms, llhs, sort_by_llh=True):
    #NOTE: modified by Jarry, March 2022
    unique = {}

    for A, L in zip(adjms, llhs):
        parents = util.convert_adjmatrix_to_parents(A)
        H = hash(parents.tobytes())
        if H in unique:
            assert np.isclose(L, unique[H]['llh'])
            assert np.array_equal(parents, unique[H]['struct'])
            unique[H]['count'] += 1
        else:
            unique[H] = {
                'struct': parents,
                'llh': L,
                'count': 1,
            }

    if sort_by_llh:
        unique = sorted(unique.values(), key = lambda T: -(np.log(T['count']) + T['llh']))
    else:
        unique = list(unique.values())
    unzipped = {key: np.array([U[key] for U in unique]) for key in unique[0].keys()}
    unzipped['prob'] = util.softmax(np.log(unzipped['count']) + unzipped['llh'])

    return (
        unzipped['struct'],
        unzipped['count'],
        unzipped['llh'],
        unzipped['prob'],
    )