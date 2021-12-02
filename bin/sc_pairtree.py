import argparse
import os
import sys
import numpy as np
import multiprocessing
import random
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import hyperparams
from util import load_sim_data
from error_rate_estimator import estimate_error_rates
from score_calculator_quad_method import calc_ancestry_tensor, complete_tensor
from tree_sampler import sample_trees
from tree_plotter import plot_tree
from pairs_tensor_plotter import plot_best_model


def _parse_args():
    #NOTE: Mostly taken from Jeff's pairtree.py, though I have commented out a few things.
    parser = argparse.ArgumentParser(
        description='Build clone trees',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
#   parser.add_argument('--verbose', action='store_true',
#     help='Print debugging messages')
    parser.add_argument('--seed', dest='seed', type=int,
        help='Integer seed used for pseudo-random number generator. Running Pairtree with the same seed on the same inputs will produce exactly the same result.')
    parser.add_argument('--parallel', dest='parallel', type=int, default=None,
        help='Number of tasks to run in parallel. By default, this is set to the number of CPU cores on the system. On hyperthreaded systems, this will be twice the number of physical CPUs.')
#   parser.add_argument('--params', dest='params_fn',
#     help='Path to JSON-formatted parameters (including mutation clusters and sample names).')
    parser.add_argument('--trees-per-chain', dest='trees_per_chain', type=int, default=3000,
        help='Total number of trees to sample in each MCMC chain.')
    parser.add_argument('--tree-chains', dest='tree_chains', type=int, default=None,
        help='Number of MCMC chains to run.')
    parser.add_argument('--burnin', dest='burnin', type=float, default=(1/3),
        help='Proportion of samples to discard from beginning of each chain.')
    parser.add_argument('--thinned-frac', dest='thinned_frac', type=float, default=1,
        help='Proportion of non-burnin trees to write as output.')
    parser.add_argument('--only-build-tensor', dest='only_build_tensor', action='store_true',
        help='Exit after building pairwise relations tensor, without sampling any trees.')
    parser.add_argument('--disable-posterior-sort', dest='sort_by_llh', action='store_false',
        help='Disable sorting posterior tree samples by descending probability, and instead list them in the order they were sampled')
    for K in hyperparams.defaults.keys():
        parser.add_argument('--%s' % K, type=float, default=hyperparams.defaults[K], help=hyperparams.explanations[K])

    parser.add_argument('data_fn')
    parser.add_argument('results_fn')
    args = parser.parse_args()
    return args

def _init_hyperparams(args):
    for K in hyperparams.defaults.keys():
        V = getattr(args, K)
        setattr(hyperparams, K, V)

def _get_default_args(args):
    # Note that multiprocessing.cpu_count() returns number of logical cores, so
    # if you're using a hyperthreaded CPU, this will be more than the number of
    # physical cores you have.
    parallel = args.parallel if args.parallel is not None else multiprocessing.cpu_count()
    if args.tree_chains is not None:
        tree_chains = args.tree_chains
    else:
        # We sometimes set `parallel = 0` to disable the use of multiprocessing,
        # making it easier to read debug messages.
        tree_chains = max(1, parallel)

    if args.seed is not None:
        seed = args.seed
    else:
        # Maximum seed is 2**32 - 1.
        seed = np.random.randint(2**32)
    return parallel, tree_chains, seed


def temp_show_it_works(adjs,llhs,accept_rates,FPR,FNR,pairs_tensor,results_fn,save_dir):
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'out', 'sc_pairtree',save_dir)
    
    with open(os.path.join(out_dir,"estimated_error_rates.txt"),'w') as f:
        f.write("FPR\tFNR\n")
        f.write(str(FPR) + '\t' + str(FNR))

    plot_best_model(pairs_tensor,outdir=out_dir,save_name="pairs_matrix.png")

    best_tree = np.argmax(llhs)
    f = plot_tree(adjs[best_tree])
    plt.savefig(os.path.join(out_dir, "best_tree.png"))


def run(data,seed):
    assert len(data.shape) == 2
    n_muts, n_cells = data.shape
    assert (n_muts > 0) & (n_cells > 0)
    print("Estimating error rates...")
    est_FPR, est_FNR = estimate_error_rates(data,subsample_cells=200,subsample_snvs=100)
    print("Calculating pairs tensor...")
    pairs_tensor = calc_ancestry_tensor(data, est_FPR, est_FNR, verbose=False, scale_integrand=True)
    pairs_tensor = complete_tensor(pairs_tensor)
    print("Sampling trees...")
    trees = sample_trees(data, pairs_tensor, FPR=est_FPR, FNR=est_FNR, 
        trees_per_chain=n_muts*500, 
        burnin=0.5, 
        nchains=4, 
        thinned_frac=0.1, 
        seed=seed, 
        parallel=4)
    return (est_FPR, est_FNR), pairs_tensor, trees


def main():
    ### PARSE ARGUMENTS ###
    args = _parse_args()
    _init_hyperparams(args)
    parallel, tree_chains, seed = _get_default_args(args)
    
    np.random.seed(seed)
    random.seed(seed)

    ### LOAD IN THE DATA ###
    data = load_sim_data(args.data_fn + "_data.txt")

    ### LOAD IN THE PARAMETERS FILE (IF I MAKE ONE) ###


    ### CREATE OBJECT WHICH CAN SAVE THE RESULTS OF THIS RUN ###


    ### ESTIMATE THE ERROR RATES ###
    print("Estimating error rates...")
    FPR, FNR = estimate_error_rates(data.data, n_iter=30, subsample_cells=200, subsample_snvs=40)

    ### CREATE THE PAIRS TENSOR ###
    print("Calculaing pairs tensor...")
    pairs_tensor = calc_ancestry_tensor(data.data, FPR, FNR, scale_integrand=True)
    pairs_tensor = complete_tensor(pairs_tensor)
    ### IF I COME UP WITH CO-CLUSTERING METHOD, INSERT HERE PROBABLY ###


    ### SAMPLE TREES ###
    print("Sampling trees...")
    adjs, llhs, accept_rates = sample_trees(data.data, pairs_tensor, FPR=FPR, FNR=FNR, 
        trees_per_chain=args.trees_per_chain, 
        burnin=args.burnin, 
        nchains=tree_chains, 
        thinned_frac=args.thinned_frac, 
        seed=seed, 
        parallel=parallel)

    ### POST-PROCESS TREES (SEE PAIRTREE.COMPUTE_POSTERIOR()) ###


    temp_show_it_works(adjs,llhs,accept_rates,FPR,FNR,pairs_tensor,args.results_fn,args.data_fn)

    return

if __name__ == "__main__":
    main()