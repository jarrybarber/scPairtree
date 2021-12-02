import os
import sys
import argparse
import numpy as np
import pickle
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
from error_rate_estimator import estimate_error_rates
from score_calculator_quad_method import calc_ancestry_tensor, complete_tensor
from tree_sampler import sample_trees
from data_simulator_full_auto import generate_simulated_data


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
    parser.add_argument('K', dest='n_clust', type=int, default=30,
        help='Number of subclones to simulate.')
    parser.add_argument('C', dest='n_cells', type=int, default=10,
        help='Number of cells per subclone.')
    parser.add_argument('M', dest='n_muts', type=int, default=10,
        help='Number of mutations per subclone.')
    parser.add_argument('A', dest='ADO', type=float, default=0.5,
        help='Allelic dropout rate.')
    parser.add_argument('P', dest='FPR', type=float, default=0.005,
        help='False positive rate.')
    parser.add_argument('data_fn')
    parser.add_argument('results_fn')
    args = parser.parse_args()
    return args

def save_data(data,true_tree,FPR,FNR,pairs_tensor,trees,save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir,"results"),'wb') as f:
        pickle.dump(data,f)
        pickle.dump(true_tree,f)
        pickle.dump(FPR,f)
        pickle.dump(FNR,f)
        pickle.dump(pairs_tensor,f)
        pickle.dump(trees,f)
    return

def main():
    args = _parse_args()
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'out', 'sims_v_sc_pairtree')

    seed = 1234
    np.random.seed(seed)
    random.seed(seed)

    
    n_cells = args.n_clust*args.n_cells_p_c
    n_muts = args.n_clust*args.n_muts_p_c
    this_save_dir = os.path.join(save_dir,"clsts{}_muts{}_cells{}_FPR{}_ADO{}".format(str(args.n_clust),str(n_muts),str(n_cells),str(args.FPR).replace('.','p'),str(args.ADO).replace('.','p')))
    data, true_tree = generate_simulated_data(args.n_clust,n_cells,n_muts,args.FPR,args.ADO)
    est_FPR, est_FNR = estimate_error_rates(data,subsample_cells=200,subsample_snvs=100)
    pairs_tensor = calc_ancestry_tensor(data, est_FPR, est_FNR, verbose=False, scale_integrand=True)
    pairs_tensor = complete_tensor(pairs_tensor)
    trees = sample_trees(data, pairs_tensor, FPR=est_FPR, FNR= est_FNR, 
        trees_per_chain=n_muts*500, 
        burnin=0.5, 
        nchains=4, 
        thinned_frac=0.1, 
        seed=seed, 
        parallel=4)
    save_data(data,true_tree,est_FPR,est_FNR,pairs_tensor,trees, this_save_dir)
                
    return

if __name__ == "__main__":
    main()