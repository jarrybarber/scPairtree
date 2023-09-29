import argparse
import os
from pickle import NONE
import sys
import numpy as np
import multiprocessing
import random
import matplotlib.pyplot as plt
import warnings
import time

from sklearn.exceptions import DataConversionWarning

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import hyperparams
from util import load_data
from error_rate_estimator import estimate_error_rates
from pairs_tensor_constructor import construct_pairs_tensor
from tree_sampler import sample_trees, compute_posterior
from tree_plotter import plot_tree
from pairs_tensor_plotter import plot_best_model
from common import DataRange, DataRangeIdx
from result_serializer import Results


def _parse_args():
    #NOTE: Mostly taken from Jeff's pairtree.py, though I have commented out a few things.
    parser = argparse.ArgumentParser(
        description='Build clone trees',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
#   parser.add_argument('--verbose', action='store_true',
#     help='Print debugging messages')
    parser.add_argument('--seed', dest='seed', type=int,
        help='Integer seed used for pseudo-random number generator. Running scPairtree with the same seed on the same inputs will produce exactly the same result.')
    parser.add_argument('--data-range', dest='d_rng_i', type=int, default=None,
        help='Data range id. There are 3 options: (0: [0,1]; 1: [0,1,3]; 2: [1,2,3])')
    parser.add_argument('--adr', dest='adr', type=float, default=None,
        help='Allelic dropout rate. If not set then will be estimated from the data.')
    parser.add_argument('--fpr', dest='fpr', type=float, default=None,
        help='False positive rate. If not set then will be estimated from the data.')
    parser.add_argument('--rerun', dest='rerun', action='store_true',
        help='Regardless of whether this datafile has already been run, rerun all analysis.')
    parser.add_argument('--variable-ado', dest='variable_ado', action='store_true',
        help='When estimating error rates, treat ADO as mutation specific. Else, ADO is treated as a global parameter for the entire dataset.')
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
    parser.add_argument('--convergence-threshold', dest='conv_thresh', type=float, default=None,
        help='(Optional) Cutoff value at which convregence will be declared and tree sampling will be terminated.')
    parser.add_argument('--convergence-min-nsamples', dest='conv_min_samp', type=float, default=None,
        help='(Optional) Minimum number of samples required before convergence criteria is checked and allowed to terminate tree sampling.')
    parser.add_argument('--check-convergence-every', dest='check_conv_every', type=float, default=None,
        help='(Optional) How often convergence is checked.')
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

    if (args.fpr is not None) or (args.adr is not None):
        if (args.fpr is not None) and (args.adr is not None):
            raise Exception("Currently, scPairtree requires either both error rates to be set or neither to be set. Please set both:\n - --adr\n - --fpr")
        
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
    
    if args.d_rng_i is None:
        d_rng_i = DataRangeIdx.ref_var_nodata
        #Might turn this into an error
        warnings.warn("Data range argument has not been specified. \n\nThis argument specifies whether the input data has known ranges: \n\t - \"no variant\" and \"variant\" [0,1] \n\t - \"reference\", \"variant\" and \"no data\" [0,1,3] \n\t - \"reference\", \" heterzygous variant\", \"homozygous variant\" and \"no data\" [0,1,2,3]. \n\nDefault is [0,1,3]")
    else:
        d_rng_i = args.d_rng_i

    if args.conv_thresh is not None or args.conv_min_samp is not None or args.check_conv_every is not None:
        if args.conv_thresh is None and args.conv_min_samp is None and args.check_conv_every is None:
            raise Exception("Not all convergence criteria is set. Either all or none of the following options should be set:\n - --convergence-threshold\n - --convergence-min-nsamples\n - --check-convergence-every")
        convergence_options = {"threshold": args.conv_thresh,
                            "min_samples": args.conv_min_samp,#int(25000*0.5*0.5),#1000,
                            "check_every": args.check_conv_every}
    else:
        convergence_options = None

    return parallel, tree_chains, seed, d_rng_i, convergence_options


def run(data, d_rng_i, variable_ado, trees_per_chain, burnin, tree_chains, thinned_frac, seed, parallel, res):
    assert len(data.shape) == 2
    n_muts, n_cells = data.shape
    ### ESTIMATE THE ERROR RATES ###
    print("Estimating error rates...")
    err_rates, _ = estimate_error_rates(data.data, d_rng_i=d_rng_i, variable_ado=variable_ado)
    FPRs, ADOs, _ = err_rates

    ### CREATE THE PAIRS TENSOR ###
    print("Constructing pairs tensor...")
    pairs_tensor = construct_pairs_tensor(data.data, FPRs, ADOs, d_rng_i=d_rng_i, scale_integrand=True)
    ### IF I COME UP WITH CO-CLUSTERING METHOD, INSERT HERE ###


    ### SAMPLE TREES ###
    print("Sampling trees...")
    adjs, llhs, accept_rates = sample_trees(data.data, pairs_tensor, FPR=FPRs, ADO=ADOs, 
        trees_per_chain=trees_per_chain, 
        burnin=burnin, 
        nchains=tree_chains, 
        thinned_frac=thinned_frac, 
        seed=seed, 
        parallel=parallel,
        d_rng_id=d_rng_i)

    return 


def main():
    ### PARSE ARGUMENTS ###
    args = _parse_args()


    ### CREATE OBJECT WHICH CAN SAVE THE RESULTS OF THIS RUN ###
    res = Results(args.results_fn)
    if res.has("scp_args") and not args.rerun:
        print("sc_pairtree already run using this results filename. Overriding current arguments with previous arguments to avoid overwriting previous results.")
        d = vars(args)
        old_args = res.get("scp_args")
        for k,v in old_args.items():
            d[k] = v
        # parallel, tree_chains, seed, d_rng_i = (old_args.parallel, old_args.tree_chains, old_args.seed, old_args.d_rng_i)
    else:
        ### SET DEFAULT ARGUMENTS ###
        args.parallel, args.tree_chains, args.seed, args.d_rng_i, convergence_options = _get_default_args(args) 
        res.add("scp_args",vars(args))
    
    # #Leaving this here temporarily so I can update some zip files lazily
    # args.parallel, args.tree_chains, args.seed, args.d_rng_i, convergence_options = _get_default_args(args) 
    # res.add("scp_args",vars(args))

    _init_hyperparams(args)
    np.random.seed(args.seed)
    random.seed(args.seed)

    
    ### LOAD IN THE DATA ###
    if res.has("data") and not args.rerun:
        data = res.get("data")
    else:
        data, gene_names = load_data(args.data_fn)
        res.add("data",data)
        res.add("gene_names", gene_names)
        res.save()
        

    ### ESTIMATE THE ERROR RATES ###
    if res.has("est_FPRs") and res.has("est_ADOs") and not args.rerun:
        fpr = res.get("est_FPRs")
        adr = res.get("est_ADOs")
    elif (args.fpr is not None) and (args.adr is not None):
        print("Error rates input, no estimation required...")
        fpr = args.fpr
        adr = args.adr
    else:
        print("Estimating error rates...")
        s = time.time()
        err_rates, _ = estimate_error_rates(data, d_rng_i=args.d_rng_i, variable_ado=args.variable_ado)
        fpr, adr, _ = err_rates
        res.add("est_runtime", time.time()-s)
        res.add("est_FPRs", fpr)
        res.add("est_ADOs", adr)
        res.save()
    

    ### CONSTRUCT THE PAIRS TENSOR ###
    if res.has("pairs_tensor") and not args.rerun:
        pairs_tensor = res.get("pairs_tensor")
    else:
        print("Constructing pairs tensor...")
        s = time.time()
        pairs_tensor = construct_pairs_tensor(data, fpr, adr, d_rng_i=args.d_rng_i, scale_integrand=True)
        res.add("pairs_tensor_runtime", time.time()-s)
        res.add("pairs_tensor", pairs_tensor)
        res.save()

    ### SAMPLE TREES ###
    if res.has("adj_mats") and not args.rerun:
        adjs = res.get("adj_mats")
        llhs = res.get("tree_llhs")
        accept_rates = res.get("accept_rates")
    else:
        print("Sampling trees...")
        s = time.time()
        best_tree, adjs, llhs, accept_rates, chain_n_samples, conv_stat = sample_trees(data, pairs_tensor, FPR=fpr, ADO=adr, 
            trees_per_chain=args.trees_per_chain, 
            burnin=args.burnin, 
            nchains=args.tree_chains, 
            thinned_frac=args.thinned_frac, 
            seed=args.seed, 
            parallel=args.parallel,
            d_rng_id=args.d_rng_i,
            convergence_options=convergence_options)
        res.add("sampling_time", time.time()-s)
        res.add("adj_mats", np.array(adjs))
        res.add("tree_llhs", np.array(llhs))
        res.add("accept_rates", np.array(accept_rates))
        res.add("best_tree_adj", np.array(best_tree.adj))
        res.add("best_tree_llh", np.array(best_tree.llh))
        res.add("chain_n_samples", np.array(chain_n_samples))
        res.add("convergence_stat", np.array(conv_stat))
        res.save()

    
    ### Compute tree posterior ###
    print("Computing tree posterior")
    post_struct, post_count, post_llh, post_prob = compute_posterior(
      adjs,
      llhs,
      True #args.sort_by_llh,
    )
    res.add('struct', post_struct)
    res.add('count', post_count)
    res.add('llh', post_llh)
    res.add('prob', post_prob)
    res.save()


    return

if __name__ == "__main__":
    main()