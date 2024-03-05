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
# import tree_util

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Make supplementary figures for sc_pairtree run',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--seed', dest='seed', type=int,
        help='Integer seed used for pseudo-random number generator. Running scPairtree with the same seed on the same inputs will produce exactly the same result.')
    parser.add_argument('--outdir', dest='outdir', default=None,
        help='Directory in which to save the figures.')
    parser.add_argument('--act-tree-adj', dest='act_tree_adj',default=None,
        help='(Optional) File containing the adjacency matrix of the actual tree, if this tree is known.')
    
    parser.add_argument('results_fn')
    args = parser.parse_args()
    return args

def _plot_err_est(fprs,ados,mut_ids,outdir):
    plt.figure()
    plt.subplot(2,1,1)
    plt.title("Estimated false positive rates")
    plt.plot(fprs)
    plt.xlabel("Muts")
    plt.ylabel("Error frequency")
    plt.xticks(np.arange(len(fprs)),mut_ids)

    plt.subplot(2,1,2)
    plt.title("Estimated allelic dropout rate")
    plt.plot(ados)
    plt.xlabel("Muts")
    plt.ylabel("Error frequency")
    plt.xticks(np.arange(len(fprs)),mut_ids)

    plt.savefig(os.path.join(outdir,"estimated_error_rates.png"))
    return


def _plot_chain_llhs(llhs,n_chains,outdir,act_tree_llh): 
    
    plt.figure(figsize=(4*n_chains,8))
    chain_lens = int(len(llhs)/n_chains)
    rng = np.max(llhs) - np.min(llhs)
    if act_tree_llh is not None:
        minmax = ( -0.05*rng + np.min(np.append(llhs,act_tree_llh)), 0.05*rng + np.max(np.append(llhs,act_tree_llh)))
    else:
        minmax = ( -0.05*rng + np.min(llhs), 0.05*rng + np.max(llhs))
    
    for i in range(n_chains):
        plt.subplot(1,n_chains,i+1)
        plt.plot(llhs[i*chain_lens:(i+1)*chain_lens])
        if act_tree_llh is not None:
            plt.plot([0, chain_lens],[act_tree_llh,act_tree_llh], 'r--')
        plt.ylim(minmax)
        plt.title("Chain {}".format(str(i)))
    
    plt.savefig(os.path.join(outdir,"chain_llhs.png"))
    return


def main():

    args = _parse_args()
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    res = Results(args.results_fn)

    scp_args = res.get('scp_args')
    data = res.get('data')
    mut_ids = res.get('mut_ids')
    est_FPRs = res.get("est_FPRs")
    est_ADOs = res.get("est_ADOs")
    est_mut_clust_ass = res.get("mutation_cluster_assignments")
    pairs_tensor = res.get("pairs_tensor")
    adjs = res.get("adj_mats")
    llhs = res.get("tree_llhs")
    accept_rates = res.get("accept_rates")
    post_struct = res.get('struct')
    post_count = res.get('count')
    post_llh = res.get('llh')
    post_prob = res.get('prob')

    best_tree_ind = np.argmax(llhs)
    best_tree_adj = adjs[best_tree_ind]

    if args.act_tree_adj is not None:
        pass
        # act_tree_adj = np.loadtxt(args.act_tree_adj, dtype=np.int16)
        # act_tree_mut_adj = tree_util.convert_nodeadj_to_mutadj(act_tree_adj, mut_assignments)
        # act_tree_anc = tree_util.convert_adjmatrix_to_ancmatrix(act_tree_adj)
        # act_tree_llh = tree_util.calc_tree_llh(data,act_tree_anc,est_FPRs,est_ADOs,scp_args['d_rng_i'])
    else:
        act_tree_adj = None
        act_tree_anc = None
        act_tree_llh = None

    plot_best_model(pairs_tensor, outdir=args.outdir, save_name="pairs_matrix.png")
    _plot_err_est(est_FPRs,est_ADOs,mut_ids,args.outdir)
    _plot_chain_llhs(llhs,scp_args['tree_chains'],args.outdir,act_tree_llh)
    fig = plot_tree(best_tree_adj)
    fig.savefig(os.path.join(args.outdir,"Highest llh tree"))

    return


if __name__ == "__main__":
    main()