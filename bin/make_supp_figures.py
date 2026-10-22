#Makes some supplementary figures for the sc_pairtree run
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from result_serializer import Results
from pairs_tensor_plotter import plot_best_model
from tree_plotter import plot_tree
# from tree_util import calc_tree_llh, convert_adjmatrix_to_ancmatrix
import tree_util

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Make supplementary figures for sc_pairtree run',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--seed', dest='seed', type=int,
        help='Integer seed used for pseudo-random number generator. Running scPairtree with the same seed on the same inputs will produce exactly the same result.')
    parser.add_argument('--outdir', dest='outdir', required=True,
        help='Directory in which to save the figures.')
    parser.add_argument('--act-tree-anc-fn', dest='act_tree_anc_fn',default=None,
        help='(Optional) File containing the ancestry matrix of the actual tree, if this tree is known.')
    parser.add_argument('--act-mut-clust-fn', dest='act_mut_clust_fn',default=None,
        help='(Optional) File containing a list of true mutation clusterings. List should be of length n_mut and each element reports which cluster a mutation belongs to.')
    parser.add_argument('--act-cell-clust-fn', dest='act_cell_clust_fn',default=None,
        help='(Optional) File containing a list of true cell clusterings. List should be of length n_cell and each element reports which cluster a cell belongs to.')
    
    parser.add_argument('results_fn')
    args = parser.parse_args()
    return args

def _plot_err_est(fprs,ados,mut_ids,outdir):
    plt.figure(figsize=(8,8))
    plt.subplot(2,1,1)
    plt.title("Estimated false positive rates")
    plt.plot(fprs)
    plt.xlabel("Muts")
    plt.ylabel("Error frequency")
    plt.xticks([])
    # plt.xticks(np.arange(len(fprs)),mut_ids)

    plt.subplot(2,1,2)
    plt.title("Estimated allelic dropout rate")
    plt.plot(ados)
    plt.xlabel("Muts")
    plt.ylabel("Error frequency")
    plt.xticks(np.arange(len(fprs)),mut_ids)

    plt.savefig(os.path.join(outdir,"estimated_error_rates.png"))
    return


def _plot_chain_llhs(llhs,chain_n_samples,outdir,act_tree_llh):
    #Chains may be of different lengths if MCMC convergence checking stopped them early
    n_chains = len(chain_n_samples)
    chain_starts = np.append(0, np.cumsum(chain_n_samples))

    plt.figure(figsize=(4*n_chains,8))
    rng = np.max(llhs) - np.min(llhs)
    if act_tree_llh is not None:
        minmax = ( -0.05*rng + np.min(np.append(llhs,act_tree_llh)), 0.05*rng + np.max(np.append(llhs,act_tree_llh)))
    else:
        minmax = ( -0.05*rng + np.min(llhs), 0.05*rng + np.max(llhs))
    
    for i in range(n_chains):
        plt.subplot(1,n_chains,i+1)
        plt.plot(llhs[chain_starts[i]:chain_starts[i+1]])
        if act_tree_llh is not None:
            plt.plot([0, chain_n_samples[i]],[act_tree_llh,act_tree_llh], 'r--')
        plt.ylim(minmax)
        plt.title("Chain {}".format(str(i)))
    
    plt.savefig(os.path.join(outdir,"chain_llhs.png"))
    return


def _plot_clustering_results(act_mut_clust_ass, est_mut_clust_ass, data, act_cell_clust_ass, outdir):
    n_mut = len(act_mut_clust_ass)
    sort_inds = np.argsort(act_mut_clust_ass)
    act_mut_clust_ass = act_mut_clust_ass[sort_inds]
    est_mut_clust_ass = est_mut_clust_ass[sort_inds]
    act_is_coclust = np.zeros((n_mut,n_mut))
    est_is_coclust = np.zeros((n_mut,n_mut))
    for i in range(n_mut):
        for j in range(n_mut):
            if act_mut_clust_ass[i] == act_mut_clust_ass[j]:
                act_is_coclust[i,j] = 1
            if est_mut_clust_ass[i] == est_mut_clust_ass[j]:
                est_is_coclust[i,j] = 1

    plt.figure(figsize=(10,15))
    plt.subplot(3,2,1)
    plt.pcolormesh(act_is_coclust)
    plt.title("actual")
    plt.subplot(3,2,3)
    plt.pcolormesh(est_is_coclust)
    plt.title("estimated")
    plt.subplot(3,2,5)
    plt.pcolormesh(act_is_coclust - est_is_coclust)
    plt.title("actual - estimated")
    
    toplt = data[sort_inds,:]
    if act_cell_clust_ass is not None:
        cell_sort_inds = np.argsort(act_cell_clust_ass)
        toplt = toplt[:,cell_sort_inds]

    plt.subplot(3,2,2)
    plt.title("data")
    plt.pcolormesh(toplt)
    plt.subplot(3,2,4)
    plt.pcolormesh(toplt)
    plt.subplot(3,2,6)
    plt.pcolormesh(toplt)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,"clustering.png"))
    

    return


def _make_clust_names(mut_clust_ass, mut_ids):
    #Set cluster names to be based on the names of the muts they're made of
    n_clusters = np.max(mut_clust_ass)
    clust_names = []
    for C in range(1,n_clusters+1):
        muts_in_clust = mut_ids[np.nonzero(mut_clust_ass==C)]
        clust_names.append( "\n".join([str(i) for i in np.sort(muts_in_clust)]) )
    return clust_names


def main():

    args = _parse_args()
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    res = Results(args.results_fn)

    scp_args = res.get('scp_arguments')
    data = res.get('data')
    n_mut, n_cell = data.shape
    mut_ids = np.asarray(res.get('mutation_ids'))
    if res.has("est_FPRs") and res.has("est_ADOs"):
        est_FPRs = res.get("est_FPRs")
        est_ADOs = res.get("est_ADOs")
    else:
        #Error rates were input rather than estimated
        est_FPRs = np.full(n_mut, scp_args['fpr'])
        est_ADOs = np.full(n_mut, scp_args['adr'])
    est_mut_clust_ass = res.get("mutation_cluster_assignments")
    pairs_tensor = res.get("pairs_tensor")
    adjs = res.get("adj_mats")
    llhs = res.get("tree_llhs")
    chain_n_samples = res.get("chain_n_samples")
    accept_rates = res.get("accept_rates")
    post_struct = res.get('struct')
    post_count = res.get('count')
    post_llh = res.get('llh')
    post_prob = res.get('prob')
    if res.has("dfpt_IS_adj_mat"):
        # dfpt_IS equivalent of a concensus graph
        dfpt_IS_adj_mat = res.get("dfpt_IS_adj_mat")

    best_tree_ind = np.argmax(llhs)
    best_tree_adj = adjs[best_tree_ind]

    if args.act_mut_clust_fn is not None:
        act_mut_clust_ass = np.loadtxt(args.act_mut_clust_fn, dtype=int)
        assert n_mut == len(act_mut_clust_ass)
    else:
        act_mut_clust_ass = None

    if args.act_tree_anc_fn is not None:
        act_tree_anc = np.loadtxt(args.act_tree_anc_fn, dtype=int)
        if act_mut_clust_ass is not None:
            act_tree_mut_ass = act_mut_clust_ass
        else:
            assert n_mut+1 == act_tree_anc.shape[0]
            act_tree_mut_ass = np.arange(1,n_mut+1)
        act_tree_llh = tree_util.calc_tree_llh(data, act_tree_anc, act_tree_mut_ass, est_FPRs, est_ADOs, scp_args['d_rng_i'])
        act_tree_adj = tree_util.convert_ancmatrix_to_adjmatrix(act_tree_anc)
        fig = plot_tree(act_tree_adj, _make_clust_names(act_tree_mut_ass, mut_ids))
        fig.savefig(os.path.join(args.outdir,"True tree"))
    else:
        act_tree_llh = None

    if args.act_cell_clust_fn is not None:
        act_cell_clust_ass = np.loadtxt(args.act_cell_clust_fn, dtype=int)
    else:
        act_cell_clust_ass = None

    plot_best_model(pairs_tensor, outdir=args.outdir, save_name="pairs_matrix.png")
    _plot_err_est(est_FPRs,est_ADOs,mut_ids,args.outdir)
    _plot_chain_llhs(llhs,chain_n_samples,args.outdir,act_tree_llh)
    fig = plot_tree(best_tree_adj, _make_clust_names(est_mut_clust_ass, mut_ids))
    fig.savefig(os.path.join(args.outdir,"Highest llh tree"))
    if act_mut_clust_ass is not None:
        _plot_clustering_results(act_mut_clust_ass, est_mut_clust_ass, data, act_cell_clust_ass, args.outdir)

    return


if __name__ == "__main__":
    main()