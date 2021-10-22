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
from score_calculator_quad_method_w_full_integrand_MLE_scaling import calc_ancestry_tensor


def plot_raw_scores(scores, outdir):
    
    scores[np.isneginf(scores)] = np.nan
    s_range = [np.nanmin(scores),np.nanmax(scores)]
    cmap = plt.get_cmap("Reds",50)
    cmap.set_bad("grey")

    plt.figure()
    plt.subplot(231)
    plt.imshow(scores[0,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 1; A-->B")
    # plt.colorbar()
    plt.subplot(232)
    plt.imshow(scores[1,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 2; B-->A")
    # plt.colorbar()
    plt.subplot(233)
    plt.imshow(scores[2,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 3; Co-clustered")
    # plt.colorbar()
    plt.subplot(234)
    plt.imshow(scores[3,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 4; Cousins")
    # plt.colorbar()
    plt.subplot(235)
    plt.imshow(scores[4,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 5; Garbage")
    # plt.colorbar()
    plt.savefig(os.path.join(outdir, "raw_scores.png"))
    plt.close()

    return

def plot_scores_differences(scores, base_scores, outdir):
    
    the_diff = (scores - base_scores)/(scores + base_scores)
    the_diff[np.isneginf(the_diff)] = np.nan
    s_range = [np.nanmin(the_diff),np.nanmax(the_diff)]
    cmap = plt.get_cmap("Reds",50)
    cmap.set_bad("grey")

    plt.figure()
    plt.subplot(231)
    plt.imshow(the_diff[0,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 1; A-->B")
    # plt.colorbar()
    plt.subplot(232)
    plt.imshow(the_diff[1,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 2; B-->A")
    # plt.colorbar()
    plt.subplot(233)
    plt.imshow(the_diff[2,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 3; Co-clustered")
    # plt.colorbar()
    plt.subplot(234)
    plt.imshow(the_diff[3,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 4; Cousins")
    # plt.colorbar()
    plt.subplot(235)
    im = plt.imshow(the_diff[4,:,:],vmin=s_range[0],vmax=s_range[1],cmap=cmap)
    plt.title("Model 5; Garbage")
    # plt.colorbar()
    cax=plt.subplot(2,30,52)
    plt.colorbar(im,cax=cax)
    plt.savefig(os.path.join(outdir, "scores_diff_from_high_accuracy_setting.png"))
    plt.close()

    return


def plot_best_model(scores, outdir, snv_ids=[]):

    best_models = np.argmax(scores,axis=0)+1
    best_models = best_models - np.tril(best_models,k=-1)

    plt.figure(figsize=(16,10))
    cMap = ListedColormap(['black', '#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00'])
    plt.imshow(best_models, vmin=-0.5, vmax=5.5, cmap=cMap)
    plt.title("Ancestry Matrix",fontsize=20)
    if len(snv_ids) > 0:
        plt.xticks(ticks=(np.linspace(0,len(snv_ids)-1,len(snv_ids))+0.5),labels=snv_ids, fontsize=12, rotation=90)
        plt.yticks(ticks=(np.linspace(0,len(snv_ids)-1,len(snv_ids))+0.5),labels=snv_ids, fontsize=12)
        plt.grid(markevery=1)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([0,1,2,3,4,5])
    cbar.ax.get_yaxis().set_ticklabels(['N/A','Y->X','X->Y','Co-incident','Branching', 'ISA Violation'],fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "best_models.png"))
    plt.close()

    return


def plot_best_nongarbage_model(scores, outdir, snv_ids):

    best_models = np.argmax(scores[0:4,:,:],axis=0)+1

    # Y = linkage(best_models, method='ward', optimal_ordering=True)
    # Z = dendrogram(Y, count_sort="descending")
    # idx = Z['leaves']
    # # idx = np.argsort(np.sum(best_models==3,axis=1))
    # best_models = best_models[idx,:]
    # best_models = best_models[:,idx]
    best_models = best_models - np.tril(best_models,k=-1)
    # snv_ids = np.array(snv_ids)[idx]

    fig, ax = plt.subplots()
    cMap = ListedColormap(['black', '#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00'])
    plt.imshow(best_models, vmin=-0.5, vmax=5.5, cmap=cMap)
    plt.title("Ancestry Matrix")
    # plt.xlabel("B")
    # plt.ylabel("A")
    if len(snv_ids) > 0:
        plt.xticks(ticks=(np.linspace(0,len(snv_ids)-1,len(snv_ids))+0.5),labels=snv_ids, fontsize=6, rotation=90)
        plt.yticks(ticks=(np.linspace(0,len(snv_ids)-1,len(snv_ids))+0.5),labels=snv_ids, fontsize=6)
        plt.grid(markevery=1)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([0,1,2,3,4,5])
    cbar.ax.get_yaxis().set_ticklabels(['N/A','Y->X','X->Y','Co-incident','Branching', 'Garbage'])
    # for j, lab in enumerate(['$A->B$','$B->A$','$Co-clustered$','$Branching$']):
    #     cbar.ax.text(1., (2 * j + 1) / 8.0, lab, ha='center', va='center')
    plt.tight_layout()
    plt.savefig(outdir + "best_nongarbage_models.png")
    plt.close()

    return


def plot_best_vs_second_best(scores, outdir):

    best_score_vs_second_best_score = np.diff(np.sort(scores,axis=0)[3:,:,:],axis=0).reshape(scores.shape[1:])
    
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    plt.imshow(-best_score_vs_second_best_score,vmin=-20,vmax=0)
    plt.title("Best vs second best model scores")
    plt.colorbar()
    plt.savefig(outdir + "best_model_score_comparisons.png")
    plt.close()

    return


def plot_raw_counts(n11,n10,n01,n00,outdir):

    fig = plt.figure()
    ax = fig.add_axes([0.05,0.55,0.4,0.4])
    plt.imshow(n11[:,:])
    plt.title("Counts: [1 1]")
    plt.colorbar()
    ax = fig.add_axes([0.55,0.55,0.4,0.4])
    plt.imshow(n10[:,:])
    plt.title("Counts: [1 0]")
    plt.colorbar()
    ax = fig.add_axes([0.05,0.05,0.4,0.4])
    plt.imshow(n01[:,:])
    plt.title("Counts: [0 1]")
    plt.colorbar()
    ax = fig.add_axes([0.55,0.05,0.4,0.4])
    plt.imshow(n00[:,:])
    plt.title("Counts: [0 0]")
    plt.colorbar()
    plt.savefig(outdir + "raw_pairwise_counts.png")
    plt.close()

    return


def plot_anc_n_cocl_comparisons(scores,outdir):

    nSSMs = scores.shape[1]
    
    M1_to_M3 = ((scores[0,:,:] - scores[2,:,:]) / scores[2,:,:]).reshape(nSSMs,nSSMs)
    M2_to_M3 = ((scores[1,:,:] - scores[2,:,:]) / scores[2,:,:]).reshape(nSSMs,nSSMs)
    range = np.max(np.abs([np.max(M1_to_M3), np.max(M2_to_M3)]))
    range = [-range, range]

    fig = plt.figure()
    ax = fig.add_axes([0.1,0.325,0.35,0.35])
    plt.imshow(M1_to_M3, cmap='seismic', vmin = range[0], vmax = range[1])
    plt.title("(M1-M3)/M3")
    plt.xlabel("B")
    plt.ylabel("A")
    plt.colorbar()
    ax = fig.add_axes([0.55,0.325,0.35,0.35])
    plt.imshow(M2_to_M3, cmap='seismic', vmin = range[0], vmax = range[1])
    plt.title("(M2-M3)/M3")
    plt.xlabel("B")
    plt.ylabel("A")
    plt.colorbar()
    plt.savefig(outdir + "anc_to_cocluster_2fig.png")
    plt.close()

    comp = ((scores[0,:,:] - scores[1,:,:])).reshape(nSSMs,nSSMs)
    fig = plt.figure()
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    plt.imshow(comp,cmap='seismic')
    plt.title("(M1-M2)")
    plt.xlabel("B")
    plt.ylabel("A")
    plt.colorbar()
    plt.savefig(outdir + "anc_to_cocluster_1fig.png")
    plt.close()
    return


def calc_anc_tensor(data, alpha, beta, tol):

    print("Determining counts of pairwise occurances...")
    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)
    
    print("Determining scores")
    scores = calc_ancestry_tensor(alpha,beta,n11,n10,n01,n00,min_tol=tol,quad_tol=tol)
    
    return scores


def main():
    sim_dat_dir = os.path.join(DATA_DIR,"simulated")

    sims_to_do = ["tree1_es1_phis1_nc20", "tree1_es1_phis1_nc40", "tree1_es1_phis1_nc100", "tree1_es2_phis1_nc20", "tree1_es2_phis1_nc40", "tree1_es2_phis1_nc100",
    "tree2_es1_phis1_nc40", "tree2_es1_phis1_nc100", "tree2_es2_phis1_nc40", "tree2_es2_phis1_nc100",
    "tree3_es1_phis1_nc100",
    "tree4_es1_phis1_nc100",
    "tree5_es1_phis1_nc395", "tree5_es2_phis1_nc395", "tree5_es3_phis1_nc395", "tree5_es4_phis1_nc395", "tree5_es5_phis1_nc395","tree5_es7_phis1_nc395",
    "tree6_es1_phis1_nc54", "tree6_es1_phis1_nc108", "tree6_es1_phis1_nc162", "tree6_es1_phis1_nc270", "tree6_es1_phis1_nc540", "tree6_es1_phis1_nc1080","tree6_es1_phis1_nc2160","tree6_es1_phis1_nc3240",
    "tree6_es2_phis1_nc54", "tree6_es2_phis1_nc108", "tree6_es2_phis1_nc162", "tree6_es2_phis1_nc270", "tree6_es2_phis1_nc540", "tree6_es2_phis1_nc1080",
    "tree6_es1_phis1_nc540", "tree6_es2_phis1_nc540", "tree6_es3_phis1_nc540", "tree6_es4_phis1_nc540", "tree6_es5_phis1_nc540",
    "tree7_es1_phis1_nc100", "tree7_es1_phis2_nc100", "tree7_es1_phis3_nc100", "tree7_es1_phis4_nc100", "tree7_es1_phis5_nc100", "tree7_es6_phis5_nc100"]

    # sims_to_do =["tree1_es1_phis1_nc100","tree1_es1_phis1_nc1000","tree1_es1_phis1_nc5000","tree1_es1_phis1_nc50000"]
    sims_to_do = ['tree1_es1_phis1_nc300']
    tol = 1e-4
    plot_comparisons = True
    for sim in sims_to_do:
        print("Working on ", sim)
        sim_data_fn = os.path.join(DATA_DIR, "simulated", sim+"_data.txt")
        # real_anc_mat_fn = os.path.join(DATA_DIR, "simulated", sim+"_ancMat.txt")
        
        outdir = os.path.join(OUT_DIR, "testing_quad_ver", "full_integrand_scaling_model_on_simulated_data", "tol_" + str(tol).replace(".","p"), sim)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        print("Loading data...")
        data, _ = load_sim_data(sim_data_fn)
        # real_anc_mat, _ = load_sim_data(real_anc_mat_fn)
        sim_tree_param = load_tree_parameters(sim,sim_dat_dir)
        alpha = sim_tree_param['alpha']
        beta  = sim_tree_param['beta']
        FP = (alpha**2*(1-beta)**2 + 2*alpha*(1-alpha)*(1-beta)**2 + 2*alpha*beta*(1-beta)) / (1-beta**2) + 0.000001
        FN = (alpha*(1-alpha)*(1-beta)**2 + beta*(1-beta)) / (1-beta**2) + 0.000001
        print("False positive rate:", FP)
        print("False negative rate:", FN)
        #Temp fix for score_calc, since it currently does not accept global error rate values.
        FP = np.zeros((data.shape[0],)) + FP
        FN = np.zeros((data.shape[0],)) + FN

        print("Data loaded.")
        
        start = time.time()        
        scores = calc_anc_tensor(data, FP, FN, tol)
        end = time.time()
        print("Calculating the anc tensor took", end-start, "seconds to complete")

        plot_best_model(scores,outdir)
        plot_raw_scores(scores,outdir)
        print("For",sim,"max score is:",np.max(scores))
        print("For",sim,"min score is:",np.min(scores))
        if plot_comparisons:
            scores_base = calc_anc_tensor(data,FP,FN,tol=1e-8)
            plot_scores_differences(scores,scores_base,outdir)

    return 





if __name__ == "__main__":
    main()