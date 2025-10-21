import argparse
import os
from pickle import NONE
import sys
import numpy as np
import multiprocessing
import random
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
# import hyperparams as hyperparams
from util import load_data
from error_rate_estimator import estimate_error_rates
from mutation_clusterer import cluster_mutations
from pairs_tensor_constructor import construct_pairs_tensor
from tree_sampler_MCMC import sample_trees, compute_posterior
from result_serializer import Results
from default_args import DEFAULT_SCP_ARGS
import tree_sampler_DFPT


def _parse_args():
    parser = argparse.ArgumentParser(
        prog='sc_pairtree',
        description="""
                    Build clone trees using single-cell genotype matrices as input. 
                    See Readme.md for more information.
                    """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    #TODO: describe required data structure for mut_id_fn
    #TODO: double check that inputting adr and fnr works as expected
    #TODO: Perhaps have default values for MCMC convergence checking, and add in a parameter that enables convergence checking.
    #TODO: Allow passing a parameter file.
    #TODO: Finish readme file.
    #TODO: Rename thinned_fraction. The name kind of implies the opposite of what the value represents (the fraction of samples that are kept).

    
    # Input and output files
    parser.add_argument('--data-fn', dest='data_fn', 
                        help='Path to the csv data file. This should contain an MxN genotype matrix, where M is the number of mutations and N is the number of cells.')
    parser.add_argument('--results-fn', dest='results_fn', 
                        help='Path to the output results file.')
    parser.add_argument('--data-range', dest='d_rng_i', type=int,
                        help='''
                            Data range identifier. This value determines the range of possible values that each element in the genotype matrix can take. 
                            This identifier can be set to either 0,1 or 2, and these correspond to the following data ranges:
                                - 0: G_ij \in {0,1} where 
                                - G_ij=0 indicates that the variant allele of locus i was not called in cell j.
                                - G_ij=1 indicates that the variant allele of locus i was called in cell j.
                                - 1: G_ij \in {0,1,3} where 
                                - G_ij=0 indicates that locus i was called as homozygous reference allele in cell j.
                                - G_ij=1 indicates that locus i was called as homozygous or heterozygous variant allele in cell j.
                                - G_ij=3 indicates that no information is available for locus i in cell j.
                                - 2: G_ij \in {0,1,2,3} where
                                - G_ij=0 indicates that locus i was called as homozygous reference allele in cell j.
                                - G_ij=1 indicates that locus i was called as heterozygous variant allele in cell j.
                                - G_ij=2 indicates that locus i was called as homozygous variant allele in cell j.
                                - G_ij=3 indicates that no information is available for locus i in cell j.
                                ''')
    parser.add_argument('--mut-id-fn', dest='mut_id_fn', type=str,
                        help='(Optional) Path to file containing identifiers for each mutation.')

    # Core options
    parser.add_argument('--seed', dest='seed', type=int, 
                        help='Integer seed used for pseudo-random number generator.')
    parser.add_argument('--parallel', dest='parallel', type=int, 
                        help='Number of tasks to run in parallel. By default, this is set to the number of CPU cores on the system. On hyperthreaded systems, this will be twice the number of physical CPUs.')

    # What-to-run options
    parser.add_argument('--resume-run', dest='resume_run', action='store_true', 
                        help='Continue a run on an existing results file. Will use arguments passed during original call while ignoring most arguments passed now. By default, erases existing files.')
    parser.add_argument('--only-estimate-errors', dest='only_estimate_errors', action='store_true', 
                        help='Exit after estimating error rates, without building pairwise relations tensor or sampling trees.')
    parser.add_argument('--only-build-tensor', dest='only_build_tensor', action='store_true',
                        help='Exit after building the pairs tensor, without sampling any trees.')
    parser.add_argument('--cluster-mutations', dest='cluster_mutations', action='store_true',
                        help='Cluster mutations and create clones trees rather than mutation trees.')
    parser.add_argument('--perform-dfpt-sampling', dest='perform_dfpt_sampling', action='store_true', 
                        help='Perform direct-from-pairs-tensor sampling. Trees sampled this way can be used with importance sampling to estimate the expected values of the consensus graph or other values.')
    parser.add_argument('--skip-mcmc-sampling', dest='skip_mcmc_tree_sampling', action='store_true', 
                        help='Skip MCMC tree sampling. Useful for testing purposes.')
    
    
    # Error rate estimation 
    parser.add_argument('--adr', dest='adr', type=float, 
                        help='Allelic dropout rate. If not set then will be estimated from the data.')
    parser.add_argument('--fpr', dest='fpr', type=float,
                        help='False positive rate. If not set then will be estimated from the data.')
    parser.add_argument('--variable-adr', dest='variable_adr', action='store_true',
                        help='When estimating error rates, treat ADR as mutation specific. Else, ADR is treated as a global parameter for the entire dataset.')

    # Mutation clustering
    parser.add_argument('--n-cluster-iter', dest='n_clust_iter', type=int,
                        help='Number of Gibbs iterations to perform during mutation clustering. Default: 20.')
    parser.add_argument('--cluster-dir-alpha', dest='clust_dir_alpha', type=float, 
                        help='Alpha paramater used in clustering for the cluster prior. Default:-0.0005 (fairly conservative)')

    # MCMC options
    parser.add_argument('--trees-per-chain', dest='trees_per_chain', type=int, 
                        help='Total number of trees to sample in each MCMC chain')
    parser.add_argument('--tree-chains', dest='tree_chains', type=int, 
                        help='Number of MCMC chains to run')
    parser.add_argument('--burnin', dest='burnin', type=float, 
                        help='Proportion of samples to discard from beginning of each chain')
    parser.add_argument('--thinned-frac', dest='thinned_frac', type=float, 
                        help='Proportion of post-burnin trees to write as output')

    # MCMC convergence options
    parser.add_argument('--convergence-threshold', dest='convergence_threshold', type=float,
                        help='(Optional) Cutoff value at which convergence will be declared and tree sampling will be terminated. Value of 1.1 is recommended.')
    parser.add_argument('--convergence-min-nsamples', dest='convergence_min_nsamples', type=int, 
                        help='(Optional) Minimum number of samples required before convergence criteria is checked and allowed to terminate tree sampling')
    parser.add_argument('--check-convergence-every', dest='check_convergence_every', type=int, 
                        help='(Optional) How often convergence is checked.')

    # DFPT sampling
    parser.add_argument('--dfpt-nsamples', dest='dfpt_nsamples', type=int,
                        help='(Optional) The number of samples to take when performing dfpt sampling. Default: 100000')

    # Hyperparameter options
    parser.add_argument('--gamma', dest='gamma', type=float,
         help="Proportion of tree modifications that should use pairs-tensor-informed choice for node to move, rather than uniform choice. Default: 0.7")
    parser.add_argument('--zeta', dest='zeta', type=float,
         help="Proportion of tree modifications that should use pairs-tensor-informed choice for destination to move node to, rather than uniform choice. Default: 0.7")
    parser.add_argument('--iota', dest='iota', type=float,
         help="Probability of initializing with pairs-tensor-informed tree rather than fully branching tree when beginning chain. Default: 0.7")

    args = parser.parse_args()
    return args

def _get_default_args(args):

    if (args.fpr is not None) or (args.adr is not None):
        if np.logical_xor((args.fpr is None), (args.adr is None)):
            raise Exception("Currently, scPairtree requires either both error rates to be set or neither to be set. Please set both:\n - --adr\n - --fpr")
    
    if args.only_estimate_errors and args.only_build_tensor:
        raise Exception("Both --only-estimate-errors and --only-build-tensor cannot both be set at the same time.")

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
        raise Exception("Data range argument has not been specified. \n\nThis argument specifies whether the input data has known ranges: \n\t - \"no variant\" and \"variant\" [0,1] \n\t - \"reference\", \"variant\" and \"no data\" [0,1,3] \n\t - \"reference\", \" heterzygous variant\", \"homozygous variant\" and \"no data\" [0,1,2,3]. \n\nPlease specify using the --data-range option.")
    else:
        d_rng_i = args.d_rng_i

    if args.conv_thresh is not None or args.conv_min_samp is not None or args.check_conv_every is not None:
        if args.conv_thresh is None and args.conv_min_samp is None and args.check_conv_every is None:
            raise Exception("Not all convergence criteria is set. Either all or none of the following options should be set:\n - --convergence-threshold\n - --convergence-min-nsamples\n - --check-convergence-every")
        convergence_options = {"threshold": args.conv_thresh,
                            "min_samples": args.conv_min_samp,
                            "check_every": args.check_conv_every}
    else:
        convergence_options = None

    

    return parallel, tree_chains, seed, d_rng_i, convergence_options


def _check_args_for_errors(args):
    """
    Checks input arguments for errors, and raises ValueError if any are found.

    Parameters:
    args (dict): Dictionary containing input arguments.

    Returns:
    None
    """

    if not type(args['seed']) == int:
        raise ValueError("Seed must be an integer.")
    if not os.path.exists(args['data_fn']):
        raise ValueError(f"Data filename {args['data_fn']} does not exist.")
    if (args['d_rng_i'] is None) or (args['d_rng_i'] not in [0,1,2]):
        raise ValueError(f"Data range index {args['d_rng_i']} is improperly set. Must be set as either 0, 1 or 2.")
    if (args['fpr'] is not None) or (args['adr'] is not None):
        if np.logical_xor((args['fpr'] is None), (args['adr'] is None)):
            raise ValueError("scPairtree requires either both error rates to be input or neither to be input. Please set both using:\n - --adr [float]\n - --fpr [float]")
    if args['only_estimate_errors'] and args['only_build_tensor']:
        raise ValueError("Both --only-estimate-errors and --only-build-tensor cannot both be set at the same time.")
    if args['only_estimate_errors'] and args['fpr'] is not None and args['adr'] is not None:
        raise ValueError("The error rates have been input with the 'only_estimate_errors' flag set. Errors are not estimated when they are input, meaning these parameters result in no tasks being performed.")
    if args["convergence_threshold"] is not None or args["convergence_min_nsamples"] is not None or args["check_convergence_every"] is not None:
        if args["convergence_threshold"] is None and args["convergence_min_nsamples"] is None and args["check_convergence_every"] is None:
                raise ValueError("""Not all MCMC convergence criteria is set. Either all or none of the following options should be set:
                                - --convergence-threshold
                                - --convergence-min-nsamples
                                - --check-convergence-every""")
    return

def _init_arguments(data_fn, d_rng_i, results_fn, input_kwargs, old_args):

    args = DEFAULT_SCP_ARGS.copy()

    if old_args is not None:
        #Use the new values of these arguments so that scPairtree can be told to do something it was told not to do previously
        args_to_keep = ['only_estimate_errors', 'only_build_tensor', 'perform_dfpt_sampling', 'skip_mcmc_tree_sampling']
        for key, value in args.items():
            if key in args_to_keep: 
                args[key] = input_kwargs.get(key, value)
                continue
            args[key] = old_args.get(key, value)
        args['data_fn'] = old_args['data_fn']
        args['d_rng_i'] = old_args['d_rng_i']
        args['results_fn'] = old_args['results_fn']
        return args
    else:
        #Overwrite default arguments with those provided by user.
        args['data_fn'] = data_fn
        args['d_rng_i'] = d_rng_i
        args['results_fn'] = results_fn
        for input_key, input_value in input_kwargs.items():
            if input_key not in args.keys():
                raise NameError(f"Did not recognize input keyword argument {input_key}. Please check spelling and refer to the Readme for valid keyword entries.")
            args[input_key] = input_value

        # ### Set non-trivial defaults ### 
        
        # Note that multiprocessing.cpu_count() returns number of logical cores, so
        # if you're using a hyperthreaded CPU, this will be more than the number of
        # physical cores you have.
        if args['parallel'] is None:
            args['parallel'] = multiprocessing.cpu_count()
        if args['tree_chains'] is None:
            # Note, sometimes set `parallel = 0` to disable the use of 
            # multiprocessing, making it easier to read debug messages.
            args['tree_chains'] = max(1, args['parallel'])

        if args['seed'] is None:
            # Maximum seed is 2**32 - 1.
            args['seed'] = np.random.randint(2**32)

        if args['convergence_threshold'] is not None or args['convergence_min_nsamples'] is not None or args['check_convergence_every'] is not None:
            args['convergence_options'] = {"threshold": args['convergence_threshold'],
                            "min_samples": args['convergence_min_nsamples'],
                            "check_every": args['check_convergence_every']}
    
    return args

# def run(data_fn, results_fn, rerun, d_rng_i, variable_adr, n_clust_iter, clust_dir_alpha, trees_per_chain, burnin, tree_chains, thinned_frac, seed, parallel, convergence_options, perform_dftp, dfpt_nsamples, fpr=[], adr=[], skip_clustering=False, only_est_errs = False, only_build_tensor = False):
def run(data_fn, d_rng_i, results_fn, **kwargs):

    ### CREATE THE RESULTS OBJECT ###
    
    # If the results file exists and want to overwrite it, then delete the old file.
    #TODO: modify this so that uses parameter "resume_run" rather than "rerun"
    if os.path.exists(results_fn) and not ('resume_run' in kwargs.keys() and kwargs['resume_run']):
        os.remove(results_fn)

    results = Results(results_fn)

    
    ### DETERMINE RUNTIME ARGUMENTS ###

    # If the results file already exists then ignore the input parameters and finish any remaining steps, if any remain.
    old_args = None
    if os.path.exists(results_fn):
        print("Resuming run...")
        old_args = results.get("scp_arguments")
    
    args = _init_arguments(data_fn, d_rng_i, results_fn, kwargs, old_args)
    _check_args_for_errors(args)

    results.add("scp_arguments", args)
    results.save()
    
    np.random.seed(args["seed"])
    random.seed(args["seed"])

    ### LOAD IN THE DATA ###
    if results.has("data"):
        data = results.get("data")
    else:
        data = load_data(args["data_fn"])
        results.add("data", data)
        results.save()
    if results.has("mutation_ids"):
        mutation_ids = results.get("mutation_ids")
    else:
        if args["mut_id_fn"] is not None:
            with open(args["mut_id_fn"], 'r') as f:
                mutation_ids = [line.strip() for line in f.readlines()]
        else:
            mutation_ids = np.arange(data.shape[0])
        results.add("mutation_ids", mutation_ids)
        results.save()
    

    ### ESTIMATE THE ERROR RATES ###
    if results.has("est_FPRs") and results.has("est_ADOs"):
        fpr = results.get("est_FPRs")
        adr = results.get("est_ADOs")
        est_mut_phis = results.get("est_mut_phis")
    elif (args["fpr"] is not None) and (args["adr"] is not None):
        print("Error rates input, no estimation required...")
        fpr, adr = args["fpr"], args["adr"]
        if np.isscalar(fpr):
            fpr = np.full(data.shape[0],fpr)
        if np.isscalar(adr):
            adr = np.full(data.shape[0],adr)
    else:
        print("Estimating error rates...")
        s = time.time()
        err_rates, _ = estimate_error_rates(data, d_rng_i=d_rng_i, variable_adr=args["variable_adr"])
        fpr, adr, est_mut_phis = err_rates
        results.add("est_runtime", time.time()-s)
        results.add("est_FPRs", fpr)
        results.add("est_ADOs", adr)
        results.add("est_mut_phis", est_mut_phis)
        results.save()
    
    if args["only_estimate_errors"]:
        return
    
    ### CLUSTER MUTATIONS ###
    if results.has("mutation_cluster_assignments"):
        mutation_cluster_assignments = results.get("mutation_cluster_assignments")
    else:
        s = time.time()
        if args["cluster_mutations"]:
            ML_mutation_cluster_assignments, all_mutation_cluster_assignments = cluster_mutations(data, fpr, adr, est_mut_phis, args['n_clust_iter'], burnin=2, pi_prior_alpha_hat=args['clust_dir_alpha'], d_rng_i=d_rng_i, ret_all_iters=True)
            mutation_cluster_assignments = all_mutation_cluster_assignments[-1]
        else:
            mutation_cluster_assignments = np.arange(1,data.shape[0]+1)
        results.add("clustering_time", time.time() - s)
        results.add("mutation_cluster_assignments", mutation_cluster_assignments)
        results.save()

    ### CONSTRUCT THE PAIRS TENSOR ###
    if results.has("pairs_tensor"):
        pairs_tensor = results.get("pairs_tensor")
    else:
        print("Constructing pairs tensor...")
        s = time.time()
        pairs_tensor = construct_pairs_tensor(data, fpr, adr, d_rng_i=d_rng_i, clst_ass=mutation_cluster_assignments-1)
        results.add("pairs_tensor_runtime", time.time()-s)
        results.add("pairs_tensor", pairs_tensor)
        results.save()
    
    if args["only_build_tensor"]:
        return

    ### SAMPLE TREES ###
    if not args["skip_mcmc_tree_sampling"]:
        if results.has("adj_mats"):
            adjs = results.get("adj_mats")
            llhs = results.get("tree_llhs")
            accept_rates = results.get("accept_rates")
        else:
            print("Sampling trees...")
            s = time.time()
            mcmc_hyperparams = {
                "iota": args['iota'],
                "gamma": args['gamma'],
                "zeta": args['zeta'],}
            best_tree, adjs, llhs, accept_rates, chain_n_samples, conv_stat = sample_trees(data, pairs_tensor, 
                mutation_cluster_assignments,
                FPR=fpr, 
                ADO=adr, 
                trees_per_chain=args['trees_per_chain'], 
                burnin=args['burnin'], 
                nchains=args['tree_chains'], 
                thinned_frac=args['thinned_frac'], 
                seed=args['seed'], 
                parallel=args['parallel'],
                d_rng_id=d_rng_i,
                hparams=mcmc_hyperparams,
                convergence_options=args['convergence_options'])
            results.add("sampling_time", time.time()-s)
            results.add("adj_mats", np.array(adjs))
            results.add("tree_llhs", np.array(llhs))
            results.add("accept_rates", np.array(accept_rates))
            results.add("best_tree_adj", np.array(best_tree.adj))
            results.add("best_tree_llh", np.array(best_tree.llh))
            results.add("chain_n_samples", np.array(chain_n_samples))
            results.add("convergence_stat", np.array(conv_stat))
            results.save()

        ### Compute tree posterior ###
        print("Computing tree posterior")
        post_struct, post_count, post_llh, post_prob = compute_posterior(
        adjs,
        llhs,
        True #args.sort_by_llh,
        )
        results.add('struct', post_struct)
        results.add('count', post_count)
        results.add('llh', post_llh)
        results.add('prob', post_prob)
        results.save()

    ### (OPTIONAL) SAMPLE DIRECTLY FROM THE PAIRS TENSOR ###
    if args["perform_dfpt_sampling"]:
        if results.has("dfpt_samples"):
            dfpt_samples = results.get("dfpt_samples")
            dfpt_sample_log_probs = results.get("dfpt_sample_log_probs")
        else:
            print("Performing direct-from-pairs-tensor sampling...")
            s = time.time()
            dfpt_samples, dfpt_sample_log_probs = tree_sampler_DFPT.sample_trees(pairs_tensor,args["dfpt_nsamples"],parallel=args["parallel"])
            # unique_samples, uniq_post, uniq_qs = dfpt_sampler.calc_uniq_samples_with_IS_posterior_prob(dfpt_samples, dfpt_sample_probs, data, fpr, adr, mutation_cluster_assignments, d_rng_i)
            log_posts = tree_sampler_DFPT.calc_sample_posts(dfpt_samples, dfpt_sample_log_probs, data, fpr, adr, mutation_cluster_assignments, d_rng_i)
            IS_adj_mat = tree_sampler_DFPT.calc_IS_adj_mat(dfpt_samples, log_posts, dfpt_sample_log_probs)
            IS_anc_mat = tree_sampler_DFPT.calc_IS_anc_mat(dfpt_samples, log_posts, dfpt_sample_log_probs)
            results.add("dfpt_time", time.time() - s)
            results.add("dfpt_samples", dfpt_samples)
            results.add("dfpt_sample_log_probs", dfpt_sample_log_probs)
            results.add("dfpt_sample_log_posteriors",log_posts)
            results.add("dfpt_IS_adj_mat", IS_adj_mat)
            results.add("dfpt_IS_anc_mat", IS_anc_mat)
            results.save()

    return 

def main():
    ### PARSE ARGUMENTS ###
    args = _parse_args()


    ### RUN SCPAIRTREE ###
    run(data_fn=args.data_fn,
        mut_id_fn=args.mut_id_fn,
        results_fn=args.results_fn,
        d_rng_i=args.d_rng_i,
        seed=args.seed,
        parallel=args.parallel,
        resume_run=args.resume_run,
        only_estimate_errors=args.only_estimate_errors,
        only_build_tensor=args.only_build_tensor,
        cluster_mutations=args.cluster_mutations,
        perform_dfpt_sampling=args.perform_dfpt_sampling,
        skip_mcmc_tree_sampling=args.skip_mcmc_tree_sampling,
        variable_adr=args.variable_adr,
        n_clust_iter=args.n_clust_iter,
        clust_dir_alpha=args.clust_dir_alpha,
        trees_per_chain=args.trees_per_chain,
        burnin=args.burnin,
        tree_chains=args.tree_chains,
        thinned_frac=args.thinned_frac,
        convergence_threshold=args.convergence_threshold,
        convergence_min_nsamples=args.convergence_min_nsamples,
        check_convergence_every=args.check_convergence_every,
        dfpt_nsamples=args.dfpt_nsamples,
        fpr=args.fpr,
        adr=args.adr,
        gamma=args.gamma,
        zeta=args.zeta,
        iota=args.iota
        )


if __name__ == "__main__":
    main()











def run_old(data, d_rng_i, variable_adr, n_clust_iter, clust_dir_alpha, trees_per_chain, burnin, tree_chains, thinned_frac, seed, parallel, convergence_options, perform_dftp, dfpt_nsamples, results, fpr=[], adr=[], skip_clustering=False, only_est_errs = False, only_build_tensor = False):
    assert len(data.shape) == 2

    ### ESTIMATE THE ERROR RATES ###
    if results.has("est_FPRs") and results.has("est_ADOs"):
        fpr = results.get("est_FPRs")
        adr = results.get("est_ADOs")
        est_mut_phis = results.get("est_mut_phis")
    elif (fpr is not None) and (adr is not None):
        print("Error rates input, no estimation required...")
        if np.isscalar(fpr):
            fpr = np.full(data.shape[0],fpr)
        if np.isscalar(adr):
            adr = np.full(data.shape[0],adr)
    else:
        print("Estimating error rates...")
        s = time.time()
        err_rates, _ = estimate_error_rates(data, d_rng_i=d_rng_i, variable_adr=variable_adr)
        fpr, adr, est_mut_phis = err_rates
        results.add("est_runtime", time.time()-s)
        results.add("est_FPRs", fpr)
        results.add("est_ADOs", adr)
        results.add("est_mut_phis", est_mut_phis)
        results.save()
    
    if only_est_errs:
        return
    
    ### CLUSTER MUTATIONS ###
    if results.has("mutation_cluster_assignments"):
        mutation_cluster_assignments = results.get("mutation_cluster_assignments")
    else:
        s = time.time()
        if skip_clustering:
            mutation_cluster_assignments = np.arange(1,data.shape[0]+1)
        else:
            ML_mutation_cluster_assignments, all_mutation_cluster_assignments = cluster_mutations(data, fpr, adr, est_mut_phis, n_clust_iter, burnin=2, pi_prior_alpha_hat=clust_dir_alpha, d_rng_i=d_rng_i, ret_all_iters=True)
            mutation_cluster_assignments = all_mutation_cluster_assignments[-1]
        results.add("clustering_time", time.time() - s)
        results.add("mutation_cluster_assignments",mutation_cluster_assignments)
        results.save()

    ### CONSTRUCT THE PAIRS TENSOR ###
    if results.has("pairs_tensor"):
        pairs_tensor = results.get("pairs_tensor")
    else:
        print("Constructing pairs tensor...")
        s = time.time()
        pairs_tensor = construct_pairs_tensor(data, fpr, adr, d_rng_i=d_rng_i, clst_ass=mutation_cluster_assignments-1)
        results.add("pairs_tensor_runtime", time.time()-s)
        results.add("pairs_tensor", pairs_tensor)
        results.save()
    
    if only_build_tensor:
        return

    ### SAMPLE TREES ###
    if results.has("adj_mats"):
        adjs = results.get("adj_mats")
        llhs = results.get("tree_llhs")
        accept_rates = results.get("accept_rates")
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
        results.add("sampling_time", time.time()-s)
        results.add("adj_mats", np.array(adjs))
        results.add("tree_llhs", np.array(llhs))
        results.add("accept_rates", np.array(accept_rates))
        results.add("best_tree_adj", np.array(best_tree.adj))
        results.add("best_tree_llh", np.array(best_tree.llh))
        results.add("chain_n_samples", np.array(chain_n_samples))
        results.add("convergence_stat", np.array(conv_stat))
        results.save()

    ### Compute tree posterior ###
    print("Computing tree posterior")
    post_struct, post_count, post_llh, post_prob = compute_posterior(
      adjs,
      llhs,
      True #args.sort_by_llh,
    )
    results.add('struct', post_struct)
    results.add('count', post_count)
    results.add('llh', post_llh)
    results.add('prob', post_prob)
    results.save()

    ### (OPTIONAL) SAMPLE DIRECTLY FROM THE PAIRS TENSOR ###
    if perform_dftp:
        if results.has("dfpt_samples"):
            dfpt_samples = results.get("dfpt_samples")
            dfpt_sample_probs = results.get("dfpt_sample_probs")
        else:
            print("Performing direct-from-pairs-tensor sampling...")
            s = time.time()
            dfpt_samples, dfpt_sample_log_probs = tree_sampler_DFPT.sample_trees(pairs_tensor,dfpt_nsamples,parallel=parallel)
            # unique_samples, uniq_post, uniq_qs = dfpt_sampler.calc_uniq_samples_with_IS_posterior_prob(dfpt_samples, dfpt_sample_probs, data, fpr, adr, mutation_cluster_assignments, d_rng_i)
            log_posts = tree_sampler_DFPT.calc_sample_posts(dfpt_samples, dfpt_sample_probs, data, fpr, adr, mutation_cluster_assignments, d_rng_i)
            IS_adj_mat = tree_sampler_DFPT.calc_IS_adj_mat(dfpt_samples, log_posts, dfpt_sample_probs)
            IS_anc_mat = tree_sampler_DFPT.calc_IS_anc_mat(dfpt_samples, log_posts, dfpt_sample_probs)
            results.add("dfpt_time", time.time() - s)
            results.add("dfpt_samples", dfpt_samples)
            results.add("dfpt_sample_log_probs", dfpt_sample_log_probs)
            results.add("dfpt_sample_log_posteriors",log_posts)
            results.add("dfpt_IS_adj_mat", IS_adj_mat)
            results.add("dfpt_IS_anc_mat", IS_anc_mat)
            results.save()

    return 


def main_old():
    ### PARSE ARGUMENTS ###
    """
    Main function for scPairtree.

    This function takes in a set of command line arguments, 
    loads in the necessary data, and runs scPairtree.

    It first checks to see if a results file already exists 
    with the same filename specified by the user. If so, it 
    will load in the arguments from the previous run and 
    use those instead of the current arguments, to avoid 
    overwriting the previous results.

    It then sets default arguments for the run, loads in 
    the data, and runs scPairtree.

    :return: None
    """
    args = _parse_args()

    if args.rerun and os.path.exists(args.results_fn):
        os.remove(args.results_fn)

    ### CREATE OBJECT WHICH CONTAINS THE RESULTS OF THIS RUN ###
    results = Results(args.results_fn)
    if results.has("sc_pairtree_args"):
        print("The results filename already exists. Overriding current arguments with previous arguments to avoid overwriting previous results.")
        old_args = results.get("sc_pairtree_args")
        for key, value in vars(args).items():
            if key == "perform_dfpt_sampling": #Don't save this value so that can run scPairtree again in the future to just perform DFPT, without having to redo the whole thing
                continue
            vars(args)[key] = old_args.get(key, value)
    else:
        ### SET DEFAULT ARGUMENTS ###
        if os.path.isfile(args.results_fn):
            os.remove(args.results_fn)
            results = Results(args.results_fn)
        args.parallel, args.tree_chains, args.seed, args.d_rng_i, args.convergence_options = _get_default_args(args)
        results.add("sc_pairtree_args", vars(args))

    _init_hyperparams(args)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ### LOAD IN THE DATA ###
    if results.has("data"):
        data = results.get("data")
    else:
        data = load_data(args.data_fn)
        results.add("data", data)
        results.save()
    if not results.has("mutation_ids"):
        if args.mutation_id_fn is not None:
            with open(args.mutation_id_fn, 'r') as f:
                mutation_ids = [line.strip() for line in f.readlines()]
        else:
            mutation_ids = np.arange(data.shape[0])
        results.add("mutation_ids", mutation_ids)
        results.save()

    ### RUN SCPAIRTREE ###
    run(data,
        d_rng_i=args.d_rng_i,
        variable_adr=args.variable_adr,
        n_clust_iter=args.n_clust_iter,
        clust_dir_alpha=args.clust_dir_alpha,
        trees_per_chain=args.trees_per_chain,
        burnin=args.burnin,
        tree_chains=args.tree_chains,
        thinned_frac=args.thinned_frac,
        seed=args.seed,
        parallel=args.parallel,
        convergence_options=args.convergence_options,
        perform_dfpt=args.perform_dfpt_sampling,
        dfpt_nsamples=args.dfpt_nsamples,
        results=results,
        fpr=args.fpr,
        adr=args.adr,
        skip_clustering=args.skip_clustering,
        only_est_errs=args.only_estimate_errors,
        only_build_tensor=args.only_build_tensor
        )