import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap

BIN_DIR = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))+ '/bin/')
sys.path.append(BIN_DIR)
from data_simulator import load_tree_parameters
from util import load_sim_data, determine_pairwise_occurance_counts
from util import DATA_DIR, OUT_DIR
from score_calculator_quad_method import calc_ancestry_tensor
from score_calculator_exact_method import calc_ancestry_tensor_exact_method




def plot_best_model_quad(scores, outdir, snv_ids=[]):

    best_models = np.argmax(scores,axis=0)+1
    best_models = best_models - np.tril(best_models,k=-1)

    plt.figure(figsize=(12,8))
    cMap = ListedColormap(['black', '#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00'])
    plt.imshow(best_models, vmin=-0.5, vmax=5.5, cmap=cMap)
    plt.title("Ancestry Matrix - Quadrature method", fontsize=20)
    plt.xlabel("X",fontdict={'fontsize':20})
    plt.ylabel("Y",fontdict={'fontsize':20})
    plt.xticks([])
    plt.yticks([])
    if len(snv_ids) > 0:
        plt.xticks(ticks=(np.linspace(0,len(snv_ids)-1,len(snv_ids))+0.5),labels=snv_ids, fontsize=12, rotation=90)
        plt.yticks(ticks=(np.linspace(0,len(snv_ids)-1,len(snv_ids))+0.5),labels=snv_ids, fontsize=12)
        plt.grid(markevery=1)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([0,1,2,3,4,5])
    cbar.ax.get_yaxis().set_ticklabels(['N/A','Y->X','X->Y','Co-incident','Branching', 'ISA Violation'],fontsize = 20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "best_models_quad.png"))
    plt.close()

    return


def plot_best_model_exact(scores, outdir, max_alpha, snv_ids=[]):

    best_models = np.argmax(scores,axis=0)+1
    best_models = best_models - np.tril(best_models,k=-1)

    plt.figure(figsize=(12,8))
    cMap = ListedColormap(['black', '#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00'])
    plt.imshow(best_models, vmin=-0.5, vmax=5.5, cmap=cMap)
    plt.title("Ancestry Matrix - Exact method", fontsize=20)
    plt.xlabel("X",fontdict={'fontsize':20})
    plt.ylabel("Y",fontdict={'fontsize':20})
    plt.xticks([])
    plt.yticks([])
    if len(snv_ids) > 0:
        plt.xticks(ticks=(np.linspace(0,len(snv_ids)-1,len(snv_ids))+0.5),labels=snv_ids, fontsize=12, rotation=90)
        plt.yticks(ticks=(np.linspace(0,len(snv_ids)-1,len(snv_ids))+0.5),labels=snv_ids, fontsize=12)
        plt.grid(markevery=1)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([0,1,2,3,4,5])
    cbar.ax.get_yaxis().set_ticklabels(['N/A','Y->X','X->Y','Co-incident','Branching', 'ISA Violation'],fontsize = 20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "best_models_exact_max_alpha_" + str(max_alpha) + ".png"))
    plt.close()

    return


def plot_differences_between_scores(scores_quad, scores_exact, outdir, max_alpha):
    
    score_diff = (np.exp(scores_quad) - np.exp(scores_exact)) / (np.exp(scores_quad) + np.exp(scores_exact))
    plt_rng = [np.min(score_diff), np.max(score_diff)]

    titles = ['Y->X','X->Y','Co-incident','Branched','Garbage']
    plt.figure(figsize=(16,8))
    for i in range(5):
        plt.subplot(2,3,1+i)
        plt.imshow(score_diff[i,:,:], vmin=plt_rng[0], vmax=plt_rng[1])
        plt.title(titles[i], fontsize=20)
        plt.xlabel("X",fontdict={'fontsize':20})
        plt.ylabel("Y",fontdict={'fontsize':20})
        plt.xticks([])
        plt.yticks([])
        cbar = plt.colorbar()
    plt.suptitle("(scores_quad - scores_exact)/(scores_quad + scores_exact)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "model_anc_tens_differences_max_alpha_" + str(max_alpha) + ".png"))
    plt.close()

    return


def calc_anc_tensor_using_quad_method(data, alpha, beta):

    print("Determining counts of pairwise occurances...")
    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)
    
    print("Determining scores")
    scores = calc_ancestry_tensor(alpha,beta,n11,n10,n01,n00)
    
    return scores


def calc_anc_tensor_using_exact_method(data, alpha, beta, max_alpha):

    print("Determining counts of pairwise occurances...")
    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)
    
    print("Determining scores")
    scores = calc_ancestry_tensor_exact_method(alpha,beta,n11,n10,n01,n00,alpha_cutoff=max_alpha, alpha_max=1)
    
    return scores


def main():
    sim_dat_dir = os.path.join(DATA_DIR,"simulated")

    # sims_to_do = ["tree1_1","tree1_2","tree1_3","tree1_4","tree1_5","tree1_6","tree1_8",'treeQ_2']
    # sims_to_do = ["sim13"]
    sims_to_do = ['tree1_8', "treeQ_2"]
    for sim in sims_to_do:
        print("Working on ", sim)
        sim_data_fn     = os.path.join(DATA_DIR, "simulated", sim+"_data.txt")
        outdir = os.path.join(OUT_DIR, "testing_quad_ver", "comparing_quad_to_exact", sim)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        print("Loading data...")
        data, _ = load_sim_data(sim_data_fn)
        sim_tree_param = load_tree_parameters(sim,sim_dat_dir)
        alpha = sim_tree_param['alpha']
        beta  = sim_tree_param['beta']
        FP = (alpha**2*(1-beta)**2 + 2*alpha*(1-alpha)*(1-beta)**2 + 2*alpha*beta*(1-beta)) / (1-beta**2)
        FN = (alpha*(1-alpha)*(1-beta)**2 + beta*(1-beta)) / (1-beta**2)
        #Temp fix for score_calc quad version, since it currently does not accept global error rate values.
        FP_vec = np.zeros((data.shape[0],)) + FP
        FN_vec = np.zeros((data.shape[0],)) + FN

        print("Data loaded.")
        
        start = time.time()        
        scores_quad = calc_anc_tensor_using_quad_method(data, FP_vec, FN_vec)
        end = time.time()
        print("Calculating the anc tensor using quad method took", end-start, "seconds to complete")

        max_alpha = 3
        start = time.time()        
        scores_exact = calc_anc_tensor_using_exact_method(data, FP, FN, max_alpha=max_alpha)
        end = time.time()
        print("Calculating the anc tensor using exact method took", end-start, "seconds to complete")

        plot_best_model_quad( scores_quad, outdir)
        plot_best_model_exact(scores_exact,outdir, max_alpha)
        plot_differences_between_scores(scores_quad, scores_exact, outdir, max_alpha)

    return 





if __name__ == "__main__":
    main()