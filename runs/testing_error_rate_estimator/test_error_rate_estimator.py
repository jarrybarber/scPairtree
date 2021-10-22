import sys
import os
import time

BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bin'))
sys.path.append(BIN_DIR)
# from data_simulator import load_tree_parameters
from util import load_sim_data, determine_pairwise_occurance_counts
from util import DATA_DIR, OUT_DIR
from data_simulator import load_tree_parameters
from error_rate_estimator import estimate_error_rates



def main():

    sim_dat_dir = os.path.join(DATA_DIR,"simulated")

    sims = ["tree1_es1_phis1_nc20", "tree1_es1_phis1_nc40", "tree1_es1_phis1_nc100", "tree1_es1_phis1_nc200", "tree1_es1_phis1_nc300",
        #es2, different nCells
    "tree1_es2_phis1_nc20", "tree1_es2_phis1_nc40", "tree1_es2_phis1_nc100", "tree1_es2_phis1_nc200", "tree1_es2_phis1_nc300",
        #es3, 300 cells, different error sets (can find any error set?)
    "tree1_es3_phis1_nc300",
    #Linear only (4 nodes):
        #es1, different nCells
    "tree2_es1_phis1_nc40","tree2_es1_phis1_nc100","tree2_es1_phis1_nc300",
        #300 cells, different error rate sets
    "tree2_es1_phis1_nc300","tree2_es2_phis1_nc300","tree2_es3_phis1_nc300",
    #Single subclone
        #300 cells, different error rate sets:
    "tree3_es1_phis1_nc300", "tree3_es2_phis1_nc300", "tree3_es3_phis1_nc300",
    #Branched only
        #2 nodes (parent node has phi=nSNV=0), 300 cells, different error rate sets:
    "tree4_es1_phis1_nc300", "tree4_es2_phis1_nc300", "tree4_es3_phis1_nc300",
        #3 nodes, first node phi=0.5, 300 cells, different error rate sets:
    "tree4_es1_phis2_nc300", "tree4_es2_phis2_nc300", "tree4_es3_phis2_nc300",
        #3 nodes, first node phi=0.0,nSNV=14, 300 cells, different error rate sets:
    "tree4_es1_phis3_nc300", "tree4_es2_phis3_nc300", "tree4_es3_phis3_nc300",
        #10ish nodes, lots of SNVs, different etas, all in branched relationships
    "tree5_es1_phis1_nc395", "tree5_es2_phis1_nc395", "tree5_es3_phis1_nc395",
    #Linear only (2 nodes):
        #100 cells, various phi combos
    "tree7_es1_phis1_nc100", "tree7_es1_phis2_nc100", "tree7_es1_phis3_nc100", "tree7_es1_phis4_nc100", "tree7_es1_phis5_nc100",
        #300 cells, es1, various phi combos
    "tree7_es1_phis1_nc300", "tree7_es1_phis2_nc300", "tree7_es1_phis3_nc300", "tree7_es1_phis4_nc300", "tree7_es1_phis5_nc300",
        #Various cells, es1, 9 pairs of nodes in linear relationships, different eta combinations
    "tree6_es1_phis1_nc54", "tree6_es1_phis1_nc108", "tree6_es1_phis1_nc162", "tree6_es1_phis1_nc270", "tree6_es1_phis1_nc540", "tree6_es1_phis1_nc1080","tree6_es1_phis1_nc2160","tree6_es1_phis1_nc3240",
        #Various cells, es2, 9 pairs of nodes in linear relationships, different eta combinations
    "tree6_es2_phis1_nc54", "tree6_es2_phis1_nc108", "tree6_es2_phis1_nc162", "tree6_es2_phis1_nc270", "tree6_es2_phis1_nc540", "tree6_es2_phis1_nc1080",
    ]
    
    # sims = ["tree6_es1_phis1_nc2160","tree6_es1_phis1_nc3240","tree6_es2_phis1_nc54", "tree6_es2_phis1_nc108", "tree6_es2_phis1_nc162", "tree6_es2_phis1_nc270", "tree6_es2_phis1_nc540", "tree6_es2_phis1_nc1080"]
    subsample_cells = 300
    subsample_snvs = 40
    for sim in sims:
        print("Working on sim:", sim)
        sim_fn = sim + '_data.txt'
        real_anc_mat_fn = sim + '_ancMat.txt'
        outdir = os.path.join(OUT_DIR,"estimating_error_rates", sim)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        data, _ = load_sim_data(sim_fn)

        sim_tree_param = load_tree_parameters(sim,sim_dat_dir)
        alpha = sim_tree_param['alpha']
        beta  = sim_tree_param['beta']
        actual_alpha = (alpha**2*(1-beta)**2 + 2*alpha*(1-alpha)*(1-beta)**2 + 2*alpha*beta*(1-beta)) / (1-beta**2)
        actual_beta = (alpha*(1-alpha)*(1-beta)**2 + beta*(1-beta)) / (1-beta**2)

        for i in range(5):
            start = time.time()
            est_alpha, est_beta = estimate_error_rates(data,subsample_cells=subsample_cells,subsample_snvs=subsample_snvs)
            end = time.time()
            print("Run took", end-start, "sec")
            print("actual FP={}, FN={}".format(actual_alpha,actual_beta))
            print("estimated FP={}, FN={}".format(est_alpha,est_beta))
            with open(os.path.join(OUT_DIR,"estimating_error_rates","estimation_results.txt"),'a') as f:
                f.write("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.2f}\n".format(sim,subsample_cells,subsample_snvs,actual_alpha,actual_beta,est_alpha,est_beta,end-start))

if __name__ == "__main__":
    main()