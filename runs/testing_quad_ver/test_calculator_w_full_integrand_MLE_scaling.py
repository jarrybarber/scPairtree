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
from score_calculator_quad_method_w_MLE_scaling import _M1_logged_integrand, _M2_logged_integrand, _M3_logged_integrand, _M4_logged_integrand, _M5_logged_integrand
from score_calculator_quad_method_w_MLE_scaling import calc_ancestry_tensor

from scipy import optimize


def plot_all_integrands_across_all_phis(data, FP, FN, phis, outdir):
    print("Determining counts of pairwise occurances...")
    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)


    #SNVs in coincident relationship
    # n11 = n11[0,1]
    # n10 = n10[0,1]
    # n01 = n01[0,1]
    # n00 = n00[0,1]

    #SNVs in linear relationship
    # n11 = n11[0,11]
    # n10 = n10[0,11]
    # n01 = n01[0,11]
    # n00 = n00[0,11]

    #SNVs in branched relationship
    n11 = n11[25,35]
    n10 = n10[25,35]
    n01 = n01[25,35]
    n00 = n00[25,35]

    M1_scores = np.zeros((len(phis),len(phis)))
    M2_scores = np.zeros((len(phis),len(phis)))
    M3_scores = np.zeros(len(phis))
    M4_scores = np.zeros((len(phis),len(phis)))
    M5_scores = np.zeros((len(phis),len(phis)))
    for i,phi_i in enumerate(phis):
        M3_scores[i] = _M3_logged_integrand(phi_i, FP, FP, FN, FN, n11, n10, n01, n00)
        for j,phi_j in enumerate(phis):
            M1_scores[i,j] = _M1_logged_integrand(phi_i, phi_j, FP, FP, FN, FN, n11, n10, n01, n00)
            M2_scores[i,j] = _M2_logged_integrand(phi_i, phi_j, FP, FP, FN, FN, n11, n10, n01, n00)
            M4_scores[i,j] = _M4_logged_integrand(phi_i, phi_j, FP, FP, FN, FN, n11, n10, n01, n00)
            M5_scores[i,j] = _M5_logged_integrand(phi_i, phi_j, FP, FP, FN, FN, n11, n10, n01, n00)
    
    M1_MLE = optimize.fmin(lambda x: -_M1_logged_integrand(x[0],x[1], FP, FP, FN, FN, n11, n10, n01, n00), [0.5,0.5])
    M2_MLE = optimize.fmin(lambda x: -_M2_logged_integrand(x[0],x[1], FP, FP, FN, FN, n11, n10, n01, n00), [0.5,0.5])
    M3_MLE = optimize.fmin(lambda x: -_M3_logged_integrand(x, FP, FP, FN, FN, n11, n10, n01, n00), 0.5, full_output=True)
    M4_MLE = optimize.fmin(lambda x: -_M4_logged_integrand(x[0],x[1], FP, FP, FN, FN, n11, n10, n01, n00), [0.5,0.5])
    M5_MLE = optimize.fmin(lambda x: -_M5_logged_integrand(x[0],x[1], FP, FP, FN, FN, n11, n10, n01, n00), [0.5,0.5])

    M3_phi_max = M3_MLE[0]
    M3_MLE = M3_MLE[1]

    M1_MLE_max_iter = optimize.fmin(lambda x: -_M1_logged_integrand(x[0],x[1], FP, FP, FN, FN, n11, n10, n01, n00), [0.5,0.5],maxiter=20)
    M2_MLE_max_iter = optimize.fmin(lambda x: -_M2_logged_integrand(x[0],x[1], FP, FP, FN, FN, n11, n10, n01, n00), [0.5,0.5],maxiter=20)
    M3_MLE_max_iter = optimize.fmin(lambda x: -_M3_logged_integrand(x, FP, FP, FN, FN, n11, n10, n01, n00), 0.5, full_output=True,maxiter=20)
    M4_MLE_max_iter = optimize.fmin(lambda x: -_M4_logged_integrand(x[0],x[1], FP, FP, FN, FN, n11, n10, n01, n00), [0.5,0.5],maxiter=20)
    M5_MLE_max_iter = optimize.fmin(lambda x: -_M5_logged_integrand(x[0],x[1], FP, FP, FN, FN, n11, n10, n01, n00), [0.5,0.5],maxiter=20)

    M3_MLE_max_iter = M3_MLE_max_iter[1]

    n11_act = n11*(1-FP)**2 + (n10+n01)*FN + n00*FN**2
    n10_act = n11*FP + n10*(1-FN-FP) + n01*FN*FP + n00*FN
    n01_act = n11*FP
    n00_act = n11*FP**2

    n = n11 + n10 + n01 + n00
    M1_MLE_my_guess = [(n11*FN*(1-FP) + n10*(FN+FP-2*FP*FN-1) + n00*(1-FN)*FP)/(n*(1-FP)*(1-FN)*(FP+FN-1)),\
                        (n01-n10)/(n*(FP+FN-1)),\
                        (n11 - FP*(n11+n10))/(n*(1-FP-FN)*(1-FN))]
    
    print(n11,n10,n01,n00)
    print( (n11+n10)/n, (n11+n01)/n)
    print ( (n11+n10)/(n11+n10+n01), (n11+n01)/(n11+n10+n01))
    print ( (n11+n10+n01)/(n11+n10+n01+n00), (n10)/(n11+n10+n01+n00))
    print ( (n11/(1-FN)**2 + n10/(1-FN))/n, (n11/(1-FN)**2+n01/(1-FN))/n)
    print ( (n11+n10)/(1-FN)/n, (n11+n01)/(1-FN)/n)
    print(M1_MLE_my_guess)

    print(M1_MLE, M1_MLE_max_iter)
    print(M2_MLE, M2_MLE_max_iter)
    print(M3_MLE, M3_MLE_max_iter)
    print(M4_MLE, M4_MLE_max_iter)
    print(M5_MLE, M5_MLE_max_iter)

    plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.title("Y->X")
    plt.pcolormesh(phis,phis,M1_scores)
    plt.plot(M1_MLE[1],M1_MLE[0],"rx")
    plt.colorbar()
    plt.subplot(232)
    plt.title("X->Y")
    plt.pcolormesh(phis,phis,M2_scores)
    plt.plot(M2_MLE[1],M2_MLE[0],"rx")
    plt.colorbar()
    plt.subplot(233)
    plt.title("Coincident")
    plt.plot(phis,M3_scores)
    plt.plot(M3_phi_max[0],-M3_MLE,"rx")
    plt.colorbar()
    plt.subplot(234)
    plt.title("Branched")
    plt.pcolormesh(phis,phis,M4_scores)
    plt.plot(M4_MLE[1],M4_MLE[0],"rx")
    plt.colorbar()
    plt.subplot(235)
    plt.title("garbage")
    plt.pcolormesh(phis,phis,M5_scores)
    plt.plot(M5_MLE[1],M5_MLE[0],"rx")
    plt.colorbar()
    plt.suptitle("n11: {}; n10: {}; n01: {}; n00".format(n11,n10,n01,n00))

    plt.savefig(os.path.join(outdir,"log(integrand).png"))
    return


def plot_resulting_anc_mats(data,FP,FN,outdir):
    print("Determining counts of pairwise occurances...")
    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)

    scores = calc_ancestry_tensor(FP,FN,n11,n10,n01,n00)

    best_models = np.argmax(scores,axis=0)+1
    best_models = best_models - np.tril(best_models,k=-1)

    plt.figure(figsize=(16,10))
    cMap = ListedColormap(['black', '#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00'])
    plt.imshow(best_models, vmin=-0.5, vmax=5.5, cmap=cMap)
    plt.title("Ancestry Matrix",fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([0,1,2,3,4,5])
    cbar.ax.get_yaxis().set_ticklabels(['N/A','Y->X','X->Y','Co-incident','Branching', 'ISA Violation'],fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "best_models.png"))
    plt.close()


    return


def main():
    sim_dat_dir = os.path.join(DATA_DIR,"simulated")
    # sim = "tree1_es1_phis1_nc100"
    sim = "tree6_es2_phis1_nc3240"

    sim_data_fn = os.path.join(DATA_DIR, "simulated", sim+"_data.txt")
    outdir = os.path.join(OUT_DIR, "testing_quad_ver", "w_MLE_scaling", sim)
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
    # FP = np.zeros((data.shape[0],)) + FP
    # FN = np.zeros((data.shape[0],)) + FN
    print("Data loaded.")


    # phis = np.linspace(0,1,50)
    # plot_all_integrands_across_all_phis(data,FP,FN,phis,outdir)

    start = time.time()
    plot_resulting_anc_mats(data,FP,FN,outdir)
    end = time.time()
    print("Took {}sec to run".format(end-start))

    return



if __name__ == "__main__":
    main()