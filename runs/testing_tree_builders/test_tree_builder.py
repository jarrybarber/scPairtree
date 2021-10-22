import numpy as np
import os
import sys

BIN_DIR = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))+ '/bin/')
sys.path.append(BIN_DIR)

from tree_builder import build_tree
from data_simulator import load_tree_parameters
from util import load_sim_data, determine_pairwise_occurance_counts
from util import DATA_DIR, OUT_DIR
from score_calculator_quad_method_w_full_integrand_MLE_scaling import calc_ancestry_tensor



def calc_anc_tens(data,FP,FN):
    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)
    anc_tens = calc_ancestry_tensor(FP,FN,n11,n10,n01,n00,)
    return anc_tens


def main():
    sim_dat_dir = os.path.join(DATA_DIR,"simulated")
    sim = "tree5_es3_phis1_nc395"

    sim_data_fn = os.path.join(DATA_DIR, "simulated", sim+"_data.txt")
    outdir = os.path.join(OUT_DIR, "testing_quad_ver", "w_split_MLE_scaling", sim)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    print("Loading data...")
    data, _ = load_sim_data(sim_data_fn)
    sim_tree_param = load_tree_parameters(sim,sim_dat_dir)

    alpha = sim_tree_param['alpha']
    beta  = sim_tree_param['beta']
    FP = (alpha**2*(1-beta)**2 + 2*alpha*(1-alpha)*(1-beta)**2 + 2*alpha*beta*(1-beta)) / (1-beta**2)
    FN = (alpha*(1-alpha)*(1-beta)**2 + beta*(1-beta)) / (1-beta**2)
    print("False positive rate:", FP)
    print("False negative rate:", FN)
    
    anc_tens = calc_anc_tens(data,FP,FN)

    build_tree(anc_tens)



    return

if __name__ == "__main__":
    main()