import argparse
import os
from pickle import NONE
import sys
import numpy as np
import multiprocessing
import random
import warnings
import time
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
import hyperparams
from util import load_data
from error_rate_estimator import estimate_error_rates
from mutation_clusterer import cluster_mutations
from pairs_tensor_constructor import construct_pairs_tensor
from tree_sampler_MCMC import sample_trees, compute_posterior
from common import DataRangeIdx
from result_serializer import Results
import tree_sampler_DFPT


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Build clone trees',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
#   parser.add_argument('--verbose', action='store_true',
#     help='Print debugging messages')
    parser.add_argument('--seed', dest='seed', type=int,
        help='Integer seed used for pseudo-random number generator. Running scPairtree with the same seed on the same inputs will produce exactly the same result.')
    parser.add_argument('--rerun', dest='rerun', action='store_true',
        help='Regardless of whether this datafile has already been run, rerun all analysis.')
    parser.add_argument('--data-range', dest='d_rng_i', type=int, default=None,
        help='Data range id. There are 3 options: (0: [0,1]; 1: [0,1,3]; 2: [0,1,2,3])')
    parser.add_argument('--adr', dest='adr', type=float, default=None,
        help='Allelic dropout rate. If not set then will be estimated from the data.')
    parser.add_argument('--fpr', dest='fpr', type=float, default=None,
        help='False positive rate. If not set then will be estimated from the data.')
    parser.add_argument('--n-cluster-iter', dest='n_clust_iter', type=int, default=100,
        help='Number of Gibbs iterations to perform during mutation clustering. Default: 100.')
    parser.add_argument('--cluster-dir-alpha', dest='clust_dir_alpha', type=float, default=1.0,
        help='Alpha paramater used in clustering for the cluster prior. Default:-2.')
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
        help='(Optional) Cutoff value at which convergence will be declared and tree sampling will be terminated.')
    parser.add_argument('--convergence-min-nsamples', dest='conv_min_samp', type=float, default=None,
        help='(Optional) Minimum number of samples required before convergence criteria is checked and allowed to terminate tree sampling.')
    parser.add_argument('--check-convergence-every', dest='check_conv_every', type=float, default=None,
        help='(Optional) How often convergence is checked.')
    parser.add_argument('--perform-dfpt-sampling', dest='perform_dfpt_sampling',  action='store_true',
        help='(Optional) Perform direct-from-pairs-tensor sampling. These can be used with importance sampling to estimate a consensus graph or other parameters.')
    parser.add_argument('--dfpt-nsamples', dest='dfpt_nsamples',  type=int, default=100000,
        help='(Optional) The number of samples to take when doing dfpt sampling.')
    parser.add_argument('--only-build-tensor', dest='only_build_tensor', action='store_true',
        help='Exit after building pairwise relations tensor, without sampling any trees.')
    parser.add_argument('--disable-posterior-sort', dest='sort_by_llh', action='store_false',
        help='Disable sorting posterior tree samples by descending probability, and instead list them in the order they were sampled')
    parser.add_argument('--mut-id-fn', dest='mut_id_fn', type=str, default=None,
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
        if np.logical_xor((args.fpr is None), (args.adr is None)):
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


def run(data, rerun, d_rng_i, variable_ado, n_clust_iter, clust_dir_alpha, trees_per_chain, burnin, tree_chains, thinned_frac, seed, parallel, convergence_options, perform_dftp, dfpt_nsamples, res, fpr=[], adr=[]):
    assert len(data.shape) == 2

    ### ESTIMATE THE ERROR RATES ###
    if res.has("est_FPRs") and res.has("est_ADOs") and not rerun:
        fpr = res.get("est_FPRs")
        adr = res.get("est_ADOs")
    elif (fpr is not None) and (adr is not None):
        print("Error rates input, no estimation required...")
        if np.isscalar(fpr):
            fpr = np.full(data.shape[0],fpr)
        if np.isscalar(adr):
            adr = np.full(data.shape[0],adr)
    else:
        print("Estimating error rates...")
        s = time.time()
        err_rates, _ = estimate_error_rates(data, d_rng_i=d_rng_i, variable_ado=variable_ado)
        fpr, adr, _ = err_rates
        res.add("est_runtime", time.time()-s)
        res.add("est_FPRs", fpr)
        res.add("est_ADOs", adr)
        res.save()
    
    ### CLUSTER MUTATIONS ###
    if res.has("mutation_cluster_assignments") and not rerun:
        mutation_cluster_assignments = res.get("mutation_cluster_assignments")
    else:
        s = time.time()
        mutation_cluster_assignments = cluster_mutations(data, fpr, adr, n_clust_iter, burnin, clust_dir_alpha, d_rng_i, ret_all_iters=False)
        res.add("clustering_time", time.time() - s)
        res.add("mutation_cluster_assignments",mutation_cluster_assignments)
        res.save()

    ### CONSTRUCT THE PAIRS TENSOR ###
    if res.has("pairs_tensor") and not rerun:
        pairs_tensor = res.get("pairs_tensor")
    else:
        print("Constructing pairs tensor...")
        s = time.time()
        pairs_tensor = construct_pairs_tensor(data, fpr, adr, d_rng_i=d_rng_i, clst_ass=mutation_cluster_assignments-1, scale_integrand=True)
        res.add("pairs_tensor_runtime", time.time()-s)
        res.add("pairs_tensor", pairs_tensor)
        res.save()

    ### SAMPLE TREES ###
    if res.has("adj_mats") and not rerun:
        adjs = res.get("adj_mats")
        llhs = res.get("tree_llhs")
        accept_rates = res.get("accept_rates")
    else:
        print("Sampling trees...")
        s = time.time()
        best_tree, adjs, llhs, accept_rates, chain_n_samples, conv_stat = sample_trees(data, pairs_tensor, 
            mutation_cluster_assignments,
            FPR=fpr, 
            ADO=adr, 
            trees_per_chain=trees_per_chain, 
            burnin=burnin, 
            nchains=tree_chains, 
            thinned_frac=thinned_frac, 
            seed=seed, 
            parallel=parallel,
            d_rng_id=d_rng_i,
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

    ### (OPTIONAL) SAMPLE DIRECTLY FROM THE PAIRS TENSOR ###
    if perform_dftp:
        if res.has("dfpt_samples") and not rerun:
            dfpt_samples = res.get("dfpt_samples")
            dfpt_sample_probs = res.get("dfpt_sample_probs")
        else:
            print("Performing direct-from-pairs-tensor sampling...")
            s = time.time()
            dfpt_samples, dfpt_sample_probs = tree_sampler_DFPT.sample_trees(pairs_tensor,dfpt_nsamples)
            # unique_samples, uniq_post, uniq_qs = dfpt_sampler.calc_uniq_samples_with_IS_posterior_prob(dfpt_samples, dfpt_sample_probs, data, fpr, adr, mutation_cluster_assignments, d_rng_i)
            log_posts = tree_sampler_DFPT.calc_sample_posts(dfpt_samples, dfpt_sample_probs, data, fpr, adr, mutation_cluster_assignments, d_rng_i)
            IS_adj_mat = tree_sampler_DFPT.calc_IS_adj_mat(dfpt_samples, log_posts, dfpt_sample_probs)
            IS_anc_mat = tree_sampler_DFPT.calc_IS_anc_mat(dfpt_samples, log_posts, dfpt_sample_probs)
            res.add("dfpt_time", time.time() - s)
            res.add("dfpt_samples", dfpt_samples)
            res.add("dfpt_sample_probs", dfpt_sample_probs)
            res.add("dfpt_IS_adj_mat", IS_adj_mat)
            res.add("dfpt_IS_anc_mat", IS_anc_mat)
            res.save()

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
    else:
        ### SET DEFAULT ARGUMENTS ###
        if os.path.isfile(args.results_fn):
            os.remove(args.results_fn)
            res = Results(args.results_fn)
        args.parallel, args.tree_chains, args.seed, args.d_rng_i, args.convergence_options = _get_default_args(args) 
        to_save_args = copy.deepcopy(args)
        to_save_args.rerun = False #Do this or setting rerun once will cause all runs to rerun in the future!
        res.add("scp_args",vars(to_save_args))
    

    _init_hyperparams(args)
    np.random.seed(args.seed)
    random.seed(args.seed)

    
    ### LOAD IN THE DATA ###
    if res.has("data") and not args.rerun:
        data = res.get("data")
    else:
        data = load_data(args.data_fn)
        res.add("data",data)
        res.save()
    if not res.has("mut_ids") or args.rerun:
        if args.mut_id_fn is not None:
            mut_ids = np.loadtxt(args.mut_id_fn)
        else:
            mut_ids = np.arange(data.shape[0])
        res.add("mut_ids",mut_ids)
        res.save()

    ### RUN SCPAIRTREE ###
    run(data,
        rerun = args.rerun,
        d_rng_i = args.d_rng_i,
        variable_ado = args.variable_ado, 
        n_clust_iter = args.n_clust_iter,
        clust_dir_alpha = args.clust_dir_alpha,
        trees_per_chain = args.trees_per_chain, 
        burnin = args.burnin, 
        tree_chains = args.tree_chains, 
        thinned_frac = args.thinned_frac, 
        seed = args.seed, 
        parallel = args.parallel, 
        convergence_options = args.convergence_options,
        perform_dftp=args.perform_dfpt_sampling,
        dfpt_nsamples=args.dfpt_nsamples,
        res = res,
        fpr = args.fpr,
        adr = args.adr
        )
        

    return

if __name__ == "__main__":
    main()