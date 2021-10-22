import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap

LIB_DIR = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))+ '/lib/')
sys.path.append(LIB_DIR)
from data_simulator import load_tree_parameters
from util import load_sim_data, determine_pairwise_occurance_counts
from util import DATA_DIR, OUT_DIR
from score_calculator_quad_method import calc_ancestry_tensor
from pairs_tensor_plotter import plot_raw_scores, plot_best_model



def main():
    sim_dat_dir = os.path.join(DATA_DIR,"simulated")

    # sims_to_do = ["tree1_es1_phis1_nc20", "tree1_es1_phis1_nc40", "tree1_es1_phis1_nc100", "tree1_es2_phis1_nc20", "tree1_es2_phis1_nc40", "tree1_es2_phis1_nc100"]
    # sims_to_do = ["tree2_es1_phis1_nc50", "tree2_es1_phis1_nc100", "tree2_es2_phis1_nc50", "tree2_es2_phis1_nc100"]
    # sims_to_do = ["tree3_es1_phis1_nc100"]
    # sims_to_do = ["tree4_es1_phis1_nc100"]
    # sims_to_do = ["tree5_es1_phis1_nc395", "tree5_es2_phis1_nc395", "tree5_es3_phis1_nc395", "tree5_es4_phis1_nc395", "tree5_es5_phis1_nc395","tree5_es7_phis1_nc395"]
    # sims_to_do = ["tree6_es1_phis1_nc54", "tree6_es1_phis1_nc108", "tree6_es1_phis1_nc162", "tree6_es1_phis1_nc270", "tree6_es1_phis1_nc540", "tree6_es1_phis1_nc1080"]
    # sims_to_do = ["tree6_es2_phis1_nc54", "tree6_es2_phis1_nc108", "tree6_es2_phis1_nc162", "tree6_es2_phis1_nc270", "tree6_es2_phis1_nc540", "tree6_es2_phis1_nc1080"]
    # sims_to_do = ["tree6_es1_phis1_nc540", "tree6_es2_phis1_nc540", "tree6_es3_phis1_nc540", "tree6_es4_phis1_nc540", "tree6_es5_phis1_nc540"]
    # sims_to_do = ["tree7_es1_phis1_nc100", "tree7_es1_phis2_nc100", "tree7_es1_phis3_nc100", "tree7_es1_phis4_nc100", "tree7_es1_phis5_nc100", "tree7_es6_phis5_nc100"]

    # sims_to_do = ["tree6_es3_phis1_nc540","tree6_es4_phis1_nc540","tree6_es5_phis1_nc540"]
    sims_to_do = ["tree1_es1_phis1_nc50000"]
    for sim in sims_to_do:
        print("Working on ", sim)
        sim_data_fn = os.path.join(DATA_DIR, "simulated", sim+"_data.txt")
        outdir = os.path.join(OUT_DIR, "testing_quad_ver", "basic_model_on_simulated_data", sim)
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
        #Temp fix for score_calc, since it currently does not accept global error rate values.
        FP = np.zeros((data.shape[0],)) + FP
        FN = np.zeros((data.shape[0],)) + FN

        print("Data loaded.")
        
        start = time.time()
        scores = calc_ancestry_tensor(data,FP,FN)
        end = time.time()
        print("Calculating the anc tensor took", end-start, "seconds to complete")

        plot_best_model(scores,outdir)
        plot_raw_scores(scores,outdir)
        print("For",sim,"max score is:",np.max(scores))
        print("For",sim,"min score is:",np.min(scores))
    return 





if __name__ == "__main__":
    main()



