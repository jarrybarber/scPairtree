import os
import sys
import argparse
import numpy as np
import pickle
import random
import multiprocessing
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from error_rate_estimator import estimate_error_rates
from score_calculator import calc_ancestry_tensor, complete_tensor
from tree_sampler import sample_trees
from data_simulator_full_auto import generate_simulated_data, _apply_errors, _put_data_in_drange_format
from common import DataRange, DataRangeIdx


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Simulate data and run through pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--seed', dest='seed', type=int,
        help='Integer seed used for pseudo-random number generator. Running Pairtree with the same seed on the same inputs will produce exactly the same result.')
    parser.add_argument('--parallel', dest='parallel', type=int, default=None,
        help='Number of tasks to run in parallel. By default, this is set to the number of CPU cores on the system. On hyperthreaded systems, this will be twice the number of physical CPUs.')
    parser.add_argument('--trees-per-chain', dest='trees_per_chain', type=int, default=3000,
        help='Total number of trees to sample in each MCMC chain.')
    parser.add_argument('--tree-chains', dest='tree_chains', type=int, default=None,
        help='Number of MCMC chains to run.')
    parser.add_argument('--burnin', dest='burnin', type=float, default=(1/3),
        help='Proportion of samples to discard from beginning of each chain.')
    parser.add_argument('--thinned-frac', dest='thinned_frac', type=float, default=1,
        help='Proportion of non-burnin trees to write as output.')
    parser.add_argument('-K', dest='n_clust', type=int, default=20,
        help='Number of subclones to simulate.')
    parser.add_argument('-C', dest='n_cell', type=int, default=200,
        help='Number of cells to simulate in dataset')
    parser.add_argument('-M', dest='n_mut', type=int, default=20,
        help='Number of mutations to simulate in dataset')
    parser.add_argument('--ado-varies', dest='ado_varies', action='store_true',
        help='Whether or not to vary the ADO across mutations.')
    parser.add_argument('-A', dest='ADO', type=float, default=0.5,
        help='Allelic dropout rate.')
    parser.add_argument('-P', dest='FPR', type=float, default=0.005,
        help='False positive rate.')
    parser.add_argument('--cell-alpha', dest='cell_alpha', type=float, default=0.5,
        help='Dirichlet distribution parameter for distributing cells to the clusters.')
    parser.add_argument('--mut-alpha', dest='mut_alpha', type=float, default=1,
        help='Dirichlet distribution parameter for distributing mutations to the clusters.')
    parser.add_argument('--data-range', dest='d_rng_id', type=int, default=DataRangeIdx.ref_var_nodata,
        help='Data range id. There are 3 options: (0: [0,1]; 1: [0,1,3]; 2: [0,1,2,3])')
    args = parser.parse_args()
    return args

def save_data(args,data,true_tree,FPR,est_ADO,ADOs,pairs_tensor,trees,runtimes,save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,"results"),'wb') as f:
        pickle.dump(args,f)
        pickle.dump(data,f)
        pickle.dump(true_tree,f)
        pickle.dump(FPR,f)
        pickle.dump(est_ADO,f)
        pickle.dump(ADOs,f)
        pickle.dump(pairs_tensor,f)
        pickle.dump(trees,f)
        pickle.dump(runtimes,f)
    return


def main():
    args = _parse_args()
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out', 'sims_v_sc_pairtree')

    assert args.n_clust <= args.n_cell
    assert args.n_clust <= args.n_mut

    if args.seed is not None:
        seed = args.seed
    else:
        # Maximum seed is 2**32 - 1.
        # seed = np.random.randint(2**32)
        seed = 1234
    np.random.seed(seed)
    random.seed(seed)

    parallel = args.parallel if args.parallel is not None else multiprocessing.cpu_count()
    if args.tree_chains is not None:
        tree_chains = args.tree_chains
    else:
        # We sometimes set `parallel = 0` to disable the use of multiprocessing,
        # making it easier to read debug messages.
        tree_chains = max(1, parallel)

    
    this_save_dir = os.path.join(save_dir,"clsts{}_muts{}_cells{}_FPR{}_ADO{}_drng_{}".format(str(args.n_clust),str(args.n_mut),str(args.n_cell),str(args.FPR).replace('.','p'),str(args.ADO).replace('.','p'),str(args.d_rng_id)))
    print("Generating data...")
    data, true_tree = generate_simulated_data(args.n_clust, 
                                                args.n_cell, 
                                                args.n_mut, 
                                                args.FPR, 
                                                args.ADO, 
                                                args.cell_alpha, 
                                                args.mut_alpha,
                                                args.d_rng_id
                                                )
    if args.ado_varies:
        error_free_data = true_tree[0]
        ADO_tightness = 20
        ADOs = np.random.beta(args.ADO*ADO_tightness, ADO_tightness*(1-args.ADO), args.n_mut)
        for i in range(args.n_mut):
            data[i,:] = _apply_errors(np.matrix(error_free_data[i,:]),args.FPR,ADOs[i])[0]
        data = _put_data_in_drange_format(data, args.d_rng_id)
    else:
        ADOs = args.ADO
    s = time.time()
    print("Estimating error rates...")
    err_rates, _ = estimate_error_rates(data)
    est_FPR, est_ADO, _ = err_rates
    e = time.time()
    rt_ER_est = e-s
    print("\tTime to estimate error rates:", rt_ER_est)
    s = time.time()
    print("Constructing pairs tensor...")
    pairs_tensor = calc_ancestry_tensor(data, est_FPR, est_ADO, verbose=False, scale_integrand=True)
    pairs_tensor = complete_tensor(pairs_tensor)
    e = time.time()
    rt_PT_calc = e-s
    s = time.time()
    print("Sampling trees...")
    trees = sample_trees(data, 
                        pairs_tensor, 
                        FPR= est_FPR, 
                        ADO= est_ADO, 
                        trees_per_chain=args.trees_per_chain, 
                        burnin=args.burnin, 
                        nchains=tree_chains, 
                        thinned_frac=args.thinned_frac, 
                        seed=seed, 
                        parallel=parallel)
    e = time.time()
    rt_sample_trees = e-s
    runtimes = {"ER_est": rt_ER_est, "pairs_tensor_calc": rt_PT_calc, "tree_sampling":rt_sample_trees}
    print("Saving data...")
    save_data(args, data, true_tree, est_FPR, est_ADO, ADOs, pairs_tensor, trees, runtimes, this_save_dir)
    print("Donzo!")
    return

if __name__ == "__main__":
    main()