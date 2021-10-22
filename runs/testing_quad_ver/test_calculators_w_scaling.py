import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap
from scipy.special import logsumexp
from scipy import integrate

BIN_DIR = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))+ '/bin/')
sys.path.append(BIN_DIR)
from data_simulator import load_tree_parameters
from util import load_sim_data, determine_pairwise_occurance_counts
from util import DATA_DIR, OUT_DIR
from score_calculator_quad_method_w_split_integrand_scaling import _get_M1_maxs, _get_M2_maxs, _get_M3_maxs, _get_M4_maxs, _get_M5_maxs
# from score_calculator_quad_method_w_split_integrand_scaling import _M1_integrand, _M2_integrand

from score_calculator_quad_method_w_full_integrand_MLE_scaling import calc_ancestry_tensor as calc_ancestry_tensor_using_intgrand_MLE
from score_calculator_quad_method_w_full_integrand_MLE_scaling import _M1_integrand
from score_calculator_quad_method_w_full_integrand_MLE_scaling import calc_score as calc_score_from_mthd
from score_calculator_util import _M1_logged_integrand, _M2_logged_integrand, _M3_logged_integrand, _M4_logged_integrand, _M5_logged_integrand
from score_calculator_util import _M1_logged_integrand_jacobian, _M2_logged_integrand_jacobian, _M3_logged_integrand_jacobian, _M4_logged_integrand_jacobian, _M5_logged_integrand_jacobian
from score_calculator_quad_method_w_split_integrand_scaling import calc_ancestry_tensor as calc_ancestry_tensor_using_term_scaled_quad
from score_calculator_quad_method_w_split_integrand_and_MLE_scaling import calc_ancestry_tensor as calc_ancestry_tensor_using_split_integrand_and_MLE_scaling_quad
from score_calculator_quad_method import calc_ancestry_tensor as calc_ancestry_tensor_using_base_quad

from scipy import optimize


#This is mostly slapdash code to test out the calculator as I wrote it...i doubt it will work at all / be useful in the future




def test_the_MLE_calcs(outdir):
    #This won't use any data or anything just check the outputs of the MLE calculators to make sure that they are giving
    #good results.
    alphas = [0.3, 0.5, 0.8]
    betas  = [0.3, 0.5, 0.8]
    phis = np.linspace(0,1,50)

    for alpha_i in alphas:
        for alpha_j in alphas:
            for beta_i in betas:
                for beta_j in betas:
                    for M in range(1,6):
                        print("a_i={}; a_j={}; b_i={}; b_j={}; M={}".format(alpha_i,alpha_j,beta_i,beta_j,M))
        
                        outdir_M = os.path.join(outdir,"M{}".format(M))
                        if not os.path.isdir(outdir_M):
                            os.makedirs(outdir_M)
                        if M == 1:
                            get_MLEs = _get_M1_MLEs
                            term1 = lambda phi_i,phi_j: (1-beta_i)*(1-beta_j)*phi_j +     (1-beta_i)*alpha_j*(phi_i - phi_j) +         alpha_i*alpha_j*(1-phi_i)
                            term2 = lambda phi_i,phi_j: (1-beta_i)*beta_j*phi_j     + (1-beta_i)*(1-alpha_j)*(phi_i - phi_j) +     alpha_i*(1-alpha_j)*(1-phi_i)
                            term3 = lambda phi_i,phi_j: beta_i*(1-beta_j)*phi_j     +         beta_i*alpha_j*(phi_i - phi_j) +     (1-alpha_i)*alpha_j*(1-phi_i)
                            term4 = lambda phi_i,phi_j: beta_i*beta_j*phi_j         +     beta_i*(1-alpha_j)*(phi_i - phi_j) + (1-alpha_i)*(1-alpha_j)*(1-phi_i)
                        elif M == 2:
                            get_MLEs = _get_M2_MLEs
                            term1 = lambda phi_i,phi_j: (1-beta_i)*(1-beta_j)*phi_i +     (1-beta_j)*alpha_i*(phi_j - phi_i) +         alpha_i*alpha_j*(1-phi_j)
                            term2 = lambda phi_i,phi_j: (1-beta_i)*beta_j*phi_i     +         beta_j*alpha_i*(phi_j - phi_i) +     alpha_i*(1-alpha_j)*(1-phi_j)
                            term3 = lambda phi_i,phi_j: beta_i*(1-beta_j)*phi_i     + (1-beta_j)*(1-alpha_i)*(phi_j - phi_i) +     (1-alpha_i)*alpha_j*(1-phi_j)
                            term4 = lambda phi_i,phi_j: beta_i*beta_j*phi_i         +     beta_j*(1-alpha_i)*(phi_j - phi_i) + (1-alpha_i)*(1-alpha_j)*(1-phi_j)
                        elif M == 3:
                            get_MLEs = _get_M3_MLEs
                            term1 = lambda phi: (1-beta_i)*(1-beta_j)*phi +         alpha_i*alpha_j*(1-phi)
                            term2 = lambda phi: (1-beta_i)*beta_j*phi     +     alpha_i*(1-alpha_j)*(1-phi)
                            term3 = lambda phi: beta_i*(1-beta_j)*phi     +     (1-alpha_i)*alpha_j*(1-phi)
                            term4 = lambda phi: beta_i*beta_j*phi         + (1-alpha_i)*(1-alpha_j)*(1-phi)
                        elif M == 4:
                            get_MLEs = _get_M4_MLEs
                            term1 = lambda phi_i,phi_j: (1-beta_i)*alpha_j*phi_i     +        alpha_i*(1-beta_j)*phi_j +         alpha_i*alpha_j*(1-phi_i-phi_j)
                            term2 = lambda phi_i,phi_j: (1-beta_i)*(1-alpha_j)*phi_i +            alpha_i*beta_j*phi_j +     alpha_i*(1-alpha_j)*(1-phi_i-phi_j)
                            term3 = lambda phi_i,phi_j: beta_i*alpha_j*phi_i         +    (1-alpha_i)*(1-beta_j)*phi_j +     (1-alpha_i)*alpha_j*(1-phi_i-phi_j)
                            term4 = lambda phi_i,phi_j: beta_i*(1-alpha_j)*phi_i     +        (1-alpha_i)*beta_j*phi_j + (1-alpha_i)*(1-alpha_j)*(1-phi_i-phi_j)
                        elif M == 5:
                            get_MLEs = _get_M5_MLEs
                            term1 = lambda phi_i,phi_j: (1-beta_i)*phi_i + alpha_i*(1-phi_i)
                            term2 = lambda phi_i,phi_j: beta_i*phi_i + (1-alpha_i)*(1-phi_i)
                            term3 = lambda phi_i,phi_j: (1-beta_j)*phi_j + alpha_j*(1-phi_j)
                            term4 = lambda phi_i,phi_j: beta_j*phi_j + (1-alpha_j)*(1-phi_j)
                        if M == 3:
                            t1_MLE, t2_MLE, t3_MLE, t4_MLE = get_MLEs(alpha_i, alpha_j, beta_i, beta_j)
                            t1_plt = [term1(phi) for phi in phis]
                            t2_plt = [term2(phi) for phi in phis]
                            t3_plt = [term3(phi) for phi in phis]
                            t4_plt = [term4(phi) for phi in phis]

                            # print("t1_MLE={}; t2_MLE={}; t3_MLE={}; t4_MLE={}".format(t1_MLE[1],t2_MLE[1],t3_MLE[1],t4_MLE[1]))

                            plt.figure(figsize=(10,10))
                            plt.suptitle("Model 3: a1={}; a2={}; b1={}; b2={}".format(alpha_i,alpha_j,beta_i,beta_j))
                            plt.subplot(221)
                            plt.plot(phis,t1_plt)
                            plt.plot(t1_MLE[0],t1_MLE[1],"r.")
                            plt.xlabel("phi")
                            plt.ylabel("term(phi)")
                            plt.title("Term 1")

                            plt.subplot(222)
                            plt.plot(phis,t2_plt)
                            plt.plot(t2_MLE[0],t2_MLE[1],"r.")
                            plt.xlabel("phi")
                            plt.ylabel("term(phi)")
                            plt.title("Term 2")

                            plt.subplot(223)
                            plt.plot(phis,t3_plt)
                            plt.plot(t3_MLE[0],t3_MLE[1],"r.")
                            plt.xlabel("phi")
                            plt.ylabel("term(phi)")
                            plt.title("Term 3")

                            plt.subplot(224)
                            plt.plot(phis,t4_plt)
                            plt.plot(t4_MLE[0],t4_MLE[1],"r.")
                            plt.xlabel("phi")
                            plt.ylabel("term(phi)")
                            plt.title("Term 4")
                            plt.savefig(os.path.join(outdir_M,"a1_{}_a2_{}_b1_{}_b2_{}.png".format(alpha_i,alpha_j,beta_i,beta_j)))
                        else:
                            t1_MLE, t2_MLE, t3_MLE, t4_MLE = get_MLEs(alpha_i, alpha_j, beta_i, beta_j)
                            t1_plt = [[term1(phi_i,phi_j) for phi_i in phis] for phi_j in phis]
                            t2_plt = [[term2(phi_i,phi_j) for phi_i in phis] for phi_j in phis]
                            t3_plt = [[term3(phi_i,phi_j) for phi_i in phis] for phi_j in phis]
                            t4_plt = [[term4(phi_i,phi_j) for phi_i in phis] for phi_j in phis]

                            # print("t1_MLE={}; t2_MLE={}; t3_MLE={}; t4_MLE={}".format(t1_MLE[2],t2_MLE[2],t3_MLE[2],t4_MLE[2]))

                            plt.figure(figsize=(10,10))
                            plt.suptitle("Model {}: a1={}; a2={}; b1={}; b2={}".format(M,alpha_i,alpha_j,beta_i,beta_j))
                            plt.subplot(221)
                            plt.pcolormesh(phis,phis,t1_plt,shading="nearest")
                            plt.contour(phis,phis,t1_plt)
                            plt.plot(t1_MLE[0],t1_MLE[1],"r.")
                            plt.xlabel("phi_i")
                            plt.ylabel("phi_j")
                            plt.title("Term 1")

                            plt.subplot(222)
                            plt.pcolormesh(phis,phis,t2_plt,shading="nearest")
                            plt.contour(phis,phis,t2_plt)
                            plt.plot(t2_MLE[0],t2_MLE[1],"r.")
                            plt.xlabel("phi_i")
                            plt.ylabel("phi_j")
                            plt.title("Term 2")

                            plt.subplot(223)
                            plt.pcolormesh(phis,phis,t3_plt,shading="nearest")
                            plt.contour(phis,phis,t3_plt)
                            plt.plot(t3_MLE[0],t3_MLE[1],"r.")
                            plt.xlabel("phi_i")
                            plt.ylabel("phi_j")
                            plt.title("Term 3")

                            plt.subplot(224)
                            plt.pcolormesh(phis,phis,t4_plt,shading="nearest")
                            plt.contour(phis,phis,t4_plt)
                            plt.plot(t4_MLE[0],t4_MLE[1],"r.")
                            plt.xlabel("phi_i")
                            plt.ylabel("phi_j")
                            plt.title("Term 4")
                            plt.savefig(os.path.join(outdir_M,"a1_{}_a2_{}_b1_{}_b2_{}.png".format(alpha_i,alpha_j,beta_i,beta_j)))
    return



def check_out_negative_score_error(outdir):
    #OOOhhhh, because I switched the phi_i and phi_j in the integrands, the hfun and gfun for models 1 and 2 were broken. Just have to fix that.
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    n11, n10, n01, n00 = (25,13,13,6)
    alpha = 0.004
    beta = 0.334
    phi_is = np.linspace(0,1,100)
    phi_js = np.linspace(0,1,100)

    for M in [1,2]:
        if M==1:
            t1_MLE, t2_MLE, t3_MLE, t4_MLE = _get_M1_MLEs(alpha, alpha, beta, beta)
            toplot_unscaled = np.array([[_M1_integrand(phi_i,phi_j,alpha,alpha,beta,beta,n11,n10,n01,n00,1,1,1,1) for phi_i in phi_is] for phi_j in phi_js ])
            toplot_scaled   = np.array([[_M1_integrand(phi_i,phi_j,alpha,alpha,beta,beta,n11,n10,n01,n00,t1_MLE[2],t2_MLE[2],t3_MLE[2],t4_MLE[2]) for phi_i in phi_is] for phi_j in phi_js ])
            toplot_unscaled = np.triu(toplot_unscaled)
            toplot_scaled = np.triu(toplot_scaled)
        else:
            t1_MLE, t2_MLE, t3_MLE, t4_MLE = _get_M2_MLEs(alpha, alpha, beta, beta)
            toplot_unscaled = np.array([[_M2_integrand(phi_i,phi_j,alpha,alpha,beta,beta,n11,n10,n01,n00,1,1,1,1) for phi_i in phi_is] for phi_j in phi_js ])
            toplot_scaled   = np.array([[_M2_integrand(phi_i,phi_j,alpha,alpha,beta,beta,n11,n10,n01,n00,t1_MLE[2],t2_MLE[2],t3_MLE[2],t4_MLE[2]) for phi_i in phi_is] for phi_j in phi_js ])
            toplot_unscaled = np.tril(toplot_unscaled)
            toplot_scaled   = np.tril(toplot_scaled)
        plt.figure(figsize=(16,8))
        plt.suptitle("Model {}; n11={}; n10={}; n01={}; n00={}".format(M, n11, n10, n01, n00))
        plt.subplot(211)
        plt.title("Unscaled")
        plt.pcolormesh(phi_is,phi_js,toplot_unscaled,shading="nearest")
        plt.colorbar()

        plt.subplot(212)
        plt.title("Scaled")
        plt.pcolormesh(phi_is,phi_js,toplot_scaled,shading="nearest")
        plt.colorbar()
        
        plt.savefig(os.path.join(outdir,"M{}.png".format(M)))

    print(_M1_integrand(0,0.8,alpha,alpha,beta,beta,n11,n10,n01,n00,1.0,1.0,1.0,1.0))
    print(_M2_integrand(0.8,0,alpha,alpha,beta,beta,n11,n10,n01,n00,1,1,1,1))

    return

def plot_resulting_anc_mats(data,FP,FN,outdir):
    print("Determining counts of pairwise occurances...")
    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)

    scores = calc_ancestry_tensor_using_split_MLE(FP,FN,n11,n10,n01,n00)

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

    plt.figure(figsize=(16,10))
    tominmax = scores + 0.0
    tominmax[np.isneginf(tominmax)] = np.nan
    score_min = np.nanmin(tominmax)
    score_max = np.nanmax(tominmax)
    plt.subplot(321)
    plt.imshow(scores[0,:,:], vmin=score_min, vmax=score_max)
    plt.title("Y->X",fontsize=15)
    plt.subplot(322)
    plt.imshow(scores[1,:,:], vmin=score_min, vmax=score_max)
    plt.title("X->Y",fontsize=15)
    plt.subplot(323)
    plt.imshow(scores[2,:,:], vmin=score_min, vmax=score_max)
    plt.title("Co-incident",fontsize=15)
    plt.subplot(324)
    plt.imshow(scores[3,:,:], vmin=score_min, vmax=score_max)
    plt.title("Branched",fontsize=15)
    plt.subplot(325)
    plt.imshow(scores[4,:,:], vmin=score_min, vmax=score_max)
    plt.title("Garbage",fontsize=15)

    plt.savefig(os.path.join(outdir, "raw_scores.png"))
    plt.close()


    return


#Just a copy from the score_calculator to see if it's faster here for some reason...
def calc_score(model,alphas,betas,n11,n10,n01,n00,verbose=True):
    
    if model==1:
        toIntegrate = _M1_integrand
        to_min      = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M1_logged_integrand(x[0],x[1],a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = np.array([0.5,0.5])
        # logged_integrand_jacobian = _M1_logged_integrand_jacobian
        L = lambda x: x #Lower bound
        U = lambda x: 1 #Upper bound
    elif model==2:
        toIntegrate = _M2_integrand
        to_min      = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M2_logged_integrand(x[0],x[1],a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = [0.5,0.5]
        # logged_integrand_jacobian = _M2_logged_integrand_jacobian
        L = lambda x: 0
        U = lambda x: x
    elif model==3:
        toIntegrate = _M3_integrand
        to_min      = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M3_logged_integrand(x,a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = 0.5
        # logged_integrand_jacobian = _M3_logged_integrand_jacobian
        L = lambda x: None
        U = lambda x: None
    elif model==4:
        toIntegrate = _M4_integrand
        to_min      = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M4_logged_integrand(x[0],x[1],a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = [0.5,0.5]
        # logged_integrand_jacobian = _M4_logged_integrand_jacobian
        L = lambda x: 0
        U = lambda x: 1-x
    elif model==5:
        toIntegrate = _M5_integrand
        to_min      = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M5_logged_integrand(x[0],x[1],a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = [0.5,0.5]
        # logged_integrand_jacobian = _M5_logged_integrand_jacobian
        L = lambda x: 0
        U = lambda x: 1
    

    nSNVs = n11.shape[0]
    assert nSNVs == len(alphas)
    assert nSNVs == len(betas)

    scores = np.zeros((nSNVs,nSNVs))
    count = 0
    f_min_times = []
    quad_times = []
    for s1 in range(nSNVs):
        for s2 in range(s1,nSNVs):
            if verbose:
                print("\r", 100.*count/(nSNVs*(nSNVs+1)/2), "% complete", end='   ')
            count += 1

            start = time.time()
            #Note: for some reason this minimize function takes ~3x longer while running here than while running in some test code (using the exact same dataset)
            # I would like to come back to this later and see if I can fix this. (Not because it's being run in parallel)
            min_res = optimize.minimize(to_min, x0, method="Nelder-Mead",tol=1e-2, args = (alphas[s1], alphas[s2], betas[s1], betas[s2], n11[s1,s2], n10[s1,s2], n01[s1,s2], n00[s1,s2]))
            logged_max = -min_res['fun']
            end = time.time()
            f_min_times.append(end-start)
            if model==3:
                start = time.time()
                score = integrate.quad(toIntegrate, 0, 1, args=(alphas[s1], alphas[s2], betas[s1], betas[s2], n11[s1,s2], n10[s1,s2], n01[s1,s2], n00[s1,s2], logged_max),epsabs=1e-2,epsrel=1e-2)
                end = time.time()
                quad_times.append(end-start)
            else:
                start = time.time()
                score = integrate.dblquad(toIntegrate, 0, 1, L, U, args=(alphas[s1], alphas[s2], betas[s1], betas[s2], n11[s1,s2], n10[s1,s2], n01[s1,s2], n00[s1,s2], logged_max),epsabs=1e-2,epsrel=1e-2)
                end = time.time()
                quad_times.append(end-start)
            scores[s1,s2] = np.log(score[0]) + logged_max
    if verbose:
        print(' ')
        print("Model", model, "runtimes:")
        print("\tavg_fmin =", np.mean(f_min_times))
        print("\tavg_quad =", np.mean(quad_times))
    return scores



def test_full_integrand_MLE_calcs(data,FP,FN,outdir):

    n11,n10,n01,n00 = determine_pairwise_occurance_counts(data)
    nSNVs = n11.shape[0]

    if np.isscalar(FP):
        alpha = np.zeros((nSNVs,)) + FP
    if np.isscalar(FN):
        beta = np.zeros((nSNVs,)) + FN
    # print("From py file:")
    # dum = calc_score_from_mthd(1,alpha,beta,n11,n10,n01,n00)
    # print("defined here")
    # dum = calc_score(1,alpha,beta,n11,n10,n01,n00)
    

    TNC_MLE = np.zeros((nSNVs,nSNVs))
    TNC_phi_is = np.zeros((nSNVs,nSNVs))
    TNC_phi_js = np.zeros((nSNVs,nSNVs))
    NM_MLE = np.zeros((nSNVs,nSNVs))
    NM_phi_is = np.zeros((nSNVs,nSNVs))
    NM_phi_js = np.zeros((nSNVs,nSNVs))
    NM_MLE_1 = np.zeros((nSNVs,nSNVs))
    NM_MLE_2 = np.zeros((nSNVs,nSNVs))
    NM_MLE_3 = np.zeros((nSNVs,nSNVs))
    NM_MLE_4 = np.zeros((nSNVs,nSNVs))
    NM_MLE_5 = np.zeros((nSNVs,nSNVs))
    NM_MLE_time1 = np.zeros((nSNVs,nSNVs)) + np.nan
    NM_MLE_time2 = np.zeros((nSNVs,nSNVs)) + np.nan
    NM_MLE_time3 = np.zeros((nSNVs,nSNVs)) + np.nan
    NM_MLE_time4 = np.zeros((nSNVs,nSNVs)) + np.nan
    NM_MLE_time5 = np.zeros((nSNVs,nSNVs)) + np.nan
    for i in range(nSNVs):
        for j in range(i,nSNVs):
            logged_integrand = lambda x: -_M1_logged_integrand(x[0],x[1],alpha[i],alpha[j],beta[i],beta[j],n11[i,j],n10[i,j],n01[i,j],n00[i,j])
            # logged_integrand_jacobian = lambda x: -_M4_logged_integrand_jacobian(x[0],x[1],FP,FP,FN,FN,n11[i,j],n10[i,j],n01[i,j],n00[i,j])
            x0 = np.array([0.5,0.5])
            # opt = optimize.minimize(logged_integrand, x0, jac = logged_integrand_jacobian, method="TNC")
            # if not opt['success']:
            #     print(opt['message'])
            # print(n11[i,j],n10[i,j],n01[i,j],n00[i,j])
            # TNC_MLE[i,j] = -opt['fun']
            # TNC_phi_is[i,j] = opt['x'][0]
            # TNC_phi_js[i,j] = opt['x'][1]
            # opt = optimize.minimize(logged_integrand, x0, method="Nelder-Mead")
            # NM_MLE[i,j] = -opt['fun']
            # NM_phi_is[i,j] = opt['x'][0]
            # NM_phi_js[i,j] = opt['x'][1]
            s = time.time(); opt = optimize.minimize(logged_integrand, x0, method="Nelder-Mead",tol=1e-6); e=time.time()
            NM_MLE_time1[i,j] = e-s
            NM_MLE_1[i,j] = -opt['fun']
            s = time.time(); opt = optimize.minimize(logged_integrand, x0, method="Nelder-Mead",tol=1e-5); e=time.time()
            NM_MLE_time2[i,j] = e-s
            NM_MLE_2[i,j] = -opt['fun']
            s = time.time(); opt = optimize.minimize(logged_integrand, x0, method="Nelder-Mead",tol=1e-4); e=time.time()
            NM_MLE_time3[i,j] = e-s
            NM_MLE_3[i,j] = -opt['fun']
            s = time.time(); opt = optimize.minimize(logged_integrand, x0, method="Nelder-Mead",tol=1e-3); e=time.time()
            NM_MLE_time4[i,j] = e-s
            NM_MLE_4[i,j] = -opt['fun']
            # dum = integrate.dblquad(_M1_integrand,0,1,lambda x:x, lambda x:1, args=(FP, FP, FN, FN, n11[i,j], n10[i,j], n01[i,j], n00[i,j], NM_MLE_4[i,j]))
            s = time.time(); opt = optimize.minimize(logged_integrand, x0, method="Nelder-Mead",tol=1e-2); e=time.time()
            NM_MLE_time5[i,j] = e-s
            NM_MLE_5[i,j] = -opt['fun']
        
    print(np.nanmean(NM_MLE_time1), np.mean(np.abs(NM_MLE_1 - NM_MLE_1)))
    print(np.nanmean(NM_MLE_time2), np.mean(np.abs(NM_MLE_1 - NM_MLE_2)))
    print(np.nanmean(NM_MLE_time3), np.mean(np.abs(NM_MLE_1 - NM_MLE_3)))
    print(np.nanmean(NM_MLE_time4), np.mean(np.abs(NM_MLE_1 - NM_MLE_4)))
    print(np.nanmean(NM_MLE_time5), np.mean(np.abs(NM_MLE_1 - NM_MLE_5)))

            
                
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.title("TNC_MLE")
    vmax = np.max([np.max(TNC_MLE), np.max(NM_MLE)])
    vmin = np.max([np.min(TNC_MLE), np.min(NM_MLE)])
    plt.pcolormesh(TNC_MLE, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.title("NM_MLE")
    vmax = np.max([np.max(TNC_MLE), np.max(NM_MLE)])
    vmin = np.max([np.min(TNC_MLE), np.min(NM_MLE)])
    plt.pcolormesh(NM_MLE, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.savefig("temp_MLE_diff_maxs")
    plt.close()

    plt.figure()
    plt.subplot(2,1,1)
    plt.title("TNC_phi_i")
    vmax = np.max([np.max(TNC_phi_is), np.max(NM_phi_is)])
    vmin = np.max([np.min(TNC_phi_is), np.min(NM_phi_is)])
    plt.pcolormesh(TNC_phi_is, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.title("NM_phi_i")
    vmax = np.max([np.max(TNC_phi_is), np.max(NM_phi_is)])
    vmin = np.max([np.min(TNC_phi_is), np.min(NM_phi_is)])
    plt.pcolormesh(NM_phi_is, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.savefig("temp_MLE_diff_phi_is")
    plt.close()

    plt.figure()
    plt.subplot(2,1,1)
    plt.title("TNC_phi_j")
    vmax = np.max([np.max(TNC_phi_js), np.max(NM_phi_js)])
    vmin = np.max([np.min(TNC_phi_js), np.min(NM_phi_js)])
    plt.pcolormesh(TNC_phi_js, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.title("NM_phi_j")
    vmax = np.max([np.max(TNC_phi_js), np.max(NM_phi_js)])
    vmin = np.max([np.min(TNC_phi_js), np.min(NM_phi_js)])
    plt.pcolormesh(NM_phi_js, vmax=vmax, vmin=vmin)
    plt.colorbar()
    plt.savefig("temp_MLE_diff_phi_js")
    plt.close()

    return
    #For just one preset mutation pair
    # n11,n10,n01,n00 = [2,140,70,160]

    phis = np.linspace(0,1,100)

    logged_integrand = lambda x: -_M5_logged_integrand(x[0],x[1],FP,FP,FN,FN,n11,n10,n01,n00)
    logged_integrand_jacobian = lambda x: -_M5_logged_integrand_jacobian(x[0],x[1],FP,FP,FN,FN,n11,n10,n01,n00)

    fmin_time = []
    wjac_time = []
    fmin_results = []
    wjac_results = []
    for i in range(1000):
        x0 = np.array([0.5,0.5]) + np.random.random(2)*0.05
        s = time.time(); fmin_max = optimize.fmin(logged_integrand,x0,disp=False); e = time.time(); fmin_time.append(e-s)
        fmin_results.append(fmin_max)
        s = time.time(); wjac_max = optimize.minimize(logged_integrand,x0,jac=logged_integrand_jacobian,method="TNC"); e = time.time(); wjac_time.append(e-s)
        wjac_results.append(wjac_max['x'])
    #So far, BFGS, TNC, SLSQP, trust-constr all work, 
    #and TNC is a real winner (works and is faster)

    #M1 seems fine
    #M2 seems fine
    #M3 seems fine
    #M4 seems fine
    #M5 seems fine
    print(np.mean(fmin_results,axis=0), np.std(fmin_results,axis=0))
    print(np.mean(wjac_results,axis=0), np.std(wjac_results,axis=0))
    print(" ")
    print(np.mean(fmin_time))
    print(np.mean(wjac_time))

    toplot = [[-logged_integrand([phi_i,phi_j]) for phi_i in phis] for phi_j in phis]

    plt.figure()
    plt.pcolormesh(phis,phis,toplot,shading="nearest")
    plt.plot(fmin_max[0],fmin_max[1],'r.')
    plt.plot(wjac_max['x'][0],wjac_max['x'][1],'gx')
    fmin_results = np.array(fmin_results)
    wjac_results = np.array(wjac_results)
    plt.plot(fmin_results[:,0],fmin_results[:,1],'r.')
    plt.plot(wjac_results[:,0],wjac_results[:,1],'gx')
    plt.savefig("temp_M5")


    #Don't forget model 3!!!
    logged_integrand = lambda x: -_M3_logged_integrand(x,FP,FP,FN,FN,n11,n10,n01,n00)
    logged_integrand_jacobian = lambda x: -_M3_logged_integrand_jacobian(x,FP,FP,FN,FN,n11,n10,n01,n00)

    fmin_time = []
    wjac_time = []
    fmin_results = []
    wjac_results = []
    for i in range(1000):
        x0 = 0.5 + np.random.random(1)*0.05
        s = time.time(); fmin_max = optimize.fmin(logged_integrand,x0,disp=False); e = time.time(); fmin_time.append(e-s)
        fmin_results.append(fmin_max)
        s = time.time(); wjac_max = optimize.minimize(logged_integrand,x0,jac=logged_integrand_jacobian,method="TNC"); e = time.time(); wjac_time.append(e-s)
        wjac_results.append(wjac_max['x'])
    #So far, BFGS, TNC, SLSQP, trust-constr all work, 
    #and TNC is a real winner (works and is faster)
    print("model 3:")
    print(np.mean(fmin_results,axis=0), np.std(fmin_results,axis=0))
    print(np.mean(wjac_results,axis=0), np.std(wjac_results,axis=0))
    print(" ")
    print(np.mean(fmin_time))
    print(np.mean(wjac_time))

    toplot = [-logged_integrand(phi) for phi in phis]

    plt.figure()
    plt.plot(phis,toplot)
    plt.plot(fmin_max,-logged_integrand(fmin_max),'r.')
    plt.plot(wjac_max['x'],-logged_integrand(wjac_max['x']),'gx')
    plt.savefig("temp_M3")


    return


def plot_the_scores(data, title, savefn):
    tominmax = data + 0.0
    tominmax[np.isneginf(tominmax)] = np.nan
    dmin = np.nanmin(tominmax)
    dmax = np.nanmax(tominmax)
    plt.figure(figsize=(12,8))
    plt.suptitle(title)
    plt.subplot(231)
    plt.title("Y->X")
    plt.pcolormesh(data[0,:,:],vmin=dmin,vmax=dmax)
    plt.subplot(232)
    plt.title("X->Y")
    plt.pcolormesh(data[1,:,:],vmin=dmin,vmax=dmax)
    plt.subplot(233)
    plt.title("Co-incident")
    plt.pcolormesh(data[2,:,:],vmin=dmin,vmax=dmax)
    plt.subplot(234)
    plt.title("Branched")
    plt.pcolormesh(data[3,:,:],vmin=dmin,vmax=dmax)
    plt.subplot(235)
    plt.title("Garbage")
    plt.pcolormesh(data[4,:,:],vmin=dmin,vmax=dmax)
    plt.colorbar()
    plt.savefig(savefn)
    plt.close()
    return

def plot_anc_mat(scores, title, savefn):
    best_models = np.argmax(scores,axis=0)+1
    best_models = best_models - np.tril(best_models,k=-1)

    plt.figure(figsize=(16,10))
    cMap = ListedColormap(['black', '#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00'])
    plt.imshow(best_models, vmin=-0.5, vmax=5.5, cmap=cMap)
    plt.title(title,fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().set_ticks([0,1,2,3,4,5])
    cbar.ax.get_yaxis().set_ticklabels(['N/A','Y->X','X->Y','Co-incident','Branching', 'ISA Violation'],fontsize=20)
    plt.tight_layout()
    plt.savefig(savefn)
    plt.close()
    return


def compare_anc_tensor_contructors(data,FP,FN,outdir):

    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)


    print("Base quad")
    start = time.time()
    scores_base_quad = calc_ancestry_tensor_using_base_quad(FP,FN,n11,n10,n01,n00)
    end = time.time()
    base_quad_time = end-start

    print("Quad with integrand MLE")
    start = time.time()
    scores_quad_integrand_MLE = calc_ancestry_tensor_using_intgrand_MLE(FP,FN,n11,n10,n01,n00)
    end = time.time()
    quad_integrand_MLE_time = end-start

    print("Quad with terms scaled by term maxes")
    start = time.time()
    scores_quad_term_scaled = calc_ancestry_tensor_using_term_scaled_quad(FP,FN,n11,n10,n01,n00)
    end = time.time()
    quad_term_scaled_time = end-start

    print("Quad with terms scaled by MLE maxes")
    start = time.time()
    scores_quad_term_MLE_scaled = calc_ancestry_tensor_using_split_integrand_and_MLE_scaling_quad(FP,FN,n11,n10,n01,n00)
    end = time.time()
    quad_term_MLE_scaled_time = end-start

    print("Time for base quad:", base_quad_time)
    print("Time for quad with integrand MLE:", quad_integrand_MLE_time)
    print("Time for quad with term scaling:", quad_term_scaled_time)
    print("Time for quad with term scaling by MLE phis:", quad_term_MLE_scaled_time)

    print("Summed non-garbage score for base quad:", np.sum(logsumexp(scores_base_quad,axis=0)))
    print("Summed non-garbage score for quad with integrand MLE:", np.sum(logsumexp(scores_quad_integrand_MLE,axis=0)))
    print("Summed non-garbage score for quad with term scaling:", np.sum(logsumexp(scores_quad_term_scaled,axis=0)))
    print("Summed non-garbage score for quad with term scaling by MLE phis:", np.sum(logsumexp(scores_quad_term_MLE_scaled,axis=0)))

    diff_with_base = (scores_quad_term_scaled - scores_base_quad) / (scores_quad_term_scaled + scores_base_quad)
    diff_with_integrand_MLE = (scores_quad_term_scaled - scores_quad_integrand_MLE) / (scores_quad_term_scaled + scores_quad_integrand_MLE)
    diff_between_term_and_MLE_termed = (scores_quad_term_scaled - scores_quad_term_MLE_scaled) / (scores_quad_term_scaled + scores_quad_term_MLE_scaled)
    
    plot_the_scores(diff_with_base,                  "(score_termed - score_base)/(score_termed + score_base)",os.path.join(outdir,"diff_with_base.png"))
    plot_the_scores(diff_with_integrand_MLE,         "(score_termed - score_iMLE)/(score_termed + score_iMLE)",os.path.join(outdir,"diff_with_integrand_MLE.png"))
    plot_the_scores(diff_between_term_and_MLE_termed,"(score_termed - score_termed_MLE)/(score_termed + score_termed_MLE)",os.path.join(outdir,"diff_with_termed_MLE.png"))

    plot_anc_mat(scores_base_quad,           "Anc mat - Quad Method - No scaling", os.path.join(outdir,"anc_mat_-_base_method.png"))
    plot_anc_mat(scores_quad_integrand_MLE,  "Anc mat - Quad Method - full integrand scaling",os.path.join(outdir,"anc_mat_-_full_integrand_scaling_method.png"))
    plot_anc_mat(scores_quad_term_scaled,    "Anc mat - Quad Method - term(max_term_phis) scaling",os.path.join(outdir,"anc_mat_-_term_max_term_scaling_method.png"))
    plot_anc_mat(scores_quad_term_MLE_scaled,"Anc mat - Quad Method - term(max_integrand_phis) scaling",os.path.join(outdir,"anc_mat_-_term_max_integrand_scaling_method.png"))

    plot_the_scores(scores_base_quad,           "Scores - Quad Method - No scaling", os.path.join(outdir,"raw_scores_-_base_method.png"))
    plot_the_scores(scores_quad_integrand_MLE,  "Scores - Quad Method - full integrand scaling",os.path.join(outdir,"raw_scorest_-_full_integrand_scaling_method.png"))
    plot_the_scores(scores_quad_term_scaled,    "Scores - Quad Method - term(max_term_phis) scaling",os.path.join(outdir,"raw_scores_-_term_max_term_scaling_method.png"))
    plot_the_scores(scores_quad_term_MLE_scaled,"Scores - Quad Method - term(max_integrand_phis) scaling",os.path.join(outdir,"raw_scores_-_term_max_integrand_scaling_method.png"))

    return


def compare_run_times_for_diff_tol_settings(data,FP,FN,outdir):
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    n11, n10, n01, n00 = determine_pairwise_occurance_counts(data)
    scores_most_accurate = calc_ancestry_tensor_using_intgrand_MLE(FP,FN,n11,n10,n01,n00,min_tol=1e-50,quad_tol=1e-10)
    scores_most_accurate[np.isneginf(scores_most_accurate)] = np.nan

    # q_tols = [1e-10, 1e-4, 1e-2, 1, 10]
    # m_tols = [1e-10, 1e-6, 1e-4, 1e-2, 1]
    # m_tols = [1e-50, 1e-10, 1, 1e10, 1e50]
    # q_tols = [1e-4, 1e-2, 1e-1, 1, 1e1]
    # m_tols = [1e-10, 1e-5, 1, 1e5, 1e10]
    q_tols = [1e-4,1e-2,1e-1,1,10]
    m_tols = [1e-10]
    times = np.zeros((len(m_tols),len(q_tols)))
    scores = np.zeros((len(m_tols),len(q_tols)))
    fmin_tols = np.zeros((len(m_tols),len(q_tols)))
    quad_tols = np.zeros((len(m_tols),len(q_tols)))
    for i,tol_fmin in enumerate(m_tols):
        for j,tol_quad in enumerate(q_tols):
            print("Quad with integrand MLE - fmin_tol={}; quad_tol={}".format(tol_fmin,tol_quad))
            start = time.time()
            scores_quad_integrand_MLE = calc_ancestry_tensor_using_intgrand_MLE(FP,FN,n11,n10,n01,n00,min_tol=tol_fmin,quad_tol=tol_quad)
            end = time.time()
            best_models = np.argmax(scores_quad_integrand_MLE,axis=0)+1
            scores_quad_integrand_MLE[np.isneginf(scores_quad_integrand_MLE)] = np.nan
            times[i,j] = end-start
            scores[i,j] = np.nansum(logsumexp(scores_quad_integrand_MLE,axis=0))
            fmin_tols[i,j] = tol_fmin
            quad_tols[i,j] = tol_quad

            best_models = best_models - np.tril(best_models,k=-1)

            plt.figure(figsize=(16,10))
            cMap = ListedColormap(['black', '#984ea3', '#377eb8', '#4daf4a', '#e41a1c', '#ff7f00'])
            plt.imshow(best_models, vmin=-0.5, vmax=5.5, cmap=cMap)
            plt.title("Ancestry Matrix",fontsize=20)
            cbar = plt.colorbar()
            cbar.ax.get_yaxis().set_ticks([0,1,2,3,4,5])
            cbar.ax.get_yaxis().set_ticklabels(['N/A','Y->X','X->Y','Co-incident','Branching', 'ISA Violation'],fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "best_models_-_qtol_{}_-_mtol_{}.png".format(tol_quad,tol_fmin)))
            plt.close()

            plt.figure(figsize=(16,10))
            score_min = np.nanmin(scores_quad_integrand_MLE)
            score_max = np.nanmax(scores_quad_integrand_MLE)
            plt.subplot(231)
            plt.imshow(scores_quad_integrand_MLE[0,:,:], vmin=score_min, vmax=score_max)
            plt.title("Y->X",fontsize=15)
            plt.subplot(232)
            plt.imshow(scores_quad_integrand_MLE[1,:,:], vmin=score_min, vmax=score_max)
            plt.title("X->Y",fontsize=15)
            plt.subplot(233)
            plt.imshow(scores_quad_integrand_MLE[2,:,:], vmin=score_min, vmax=score_max)
            plt.title("Co-incident",fontsize=15)
            plt.subplot(234)
            plt.imshow(scores_quad_integrand_MLE[3,:,:], vmin=score_min, vmax=score_max)
            plt.title("Branched",fontsize=15)
            plt.subplot(235)
            im=plt.imshow(scores_quad_integrand_MLE[4,:,:], vmin=score_min, vmax=score_max)
            plt.title("Garbage",fontsize=15)
            cax=plt.subplot(2,30,52)
            plt.colorbar(im,cax=cax)
            plt.savefig(os.path.join(outdir, "raw_scores_-_qtol_{}_-_mtol_{}.png".format(tol_quad,tol_fmin)))
            plt.close()

            plt.figure(figsize=(16,10))
            toplot = scores_quad_integrand_MLE - scores_most_accurate
            score_min = np.nanmin(toplot)
            score_max = np.nanmax(toplot)
            plt.suptitle("Difference from 'most accurate' setting ancestry tensor")
            plt.subplot(231)
            plt.imshow(toplot[0,:,:], vmin=score_min, vmax=score_max)
            plt.title("Y->X",fontsize=15)
            plt.subplot(232)
            plt.imshow(toplot[1,:,:], vmin=score_min, vmax=score_max)
            plt.title("X->Y",fontsize=15)
            plt.subplot(233)
            plt.imshow(toplot[2,:,:], vmin=score_min, vmax=score_max)
            plt.title("Co-incident",fontsize=15)
            plt.subplot(234)
            plt.imshow(toplot[3,:,:], vmin=score_min, vmax=score_max)
            plt.title("Branched",fontsize=15)
            plt.subplot(235)
            im=plt.imshow(toplot[4,:,:], vmin=score_min, vmax=score_max)
            plt.title("Garbage",fontsize=15)
            cax=plt.subplot(2,30,52)
            plt.colorbar(im,cax=cax)
            plt.savefig(os.path.join(outdir, "scores_diff_with_accurate_setting_-_qtol_{}_-_mtol_{}.png".format(tol_quad,tol_fmin)))
            plt.close()

    print(fmin_tols)
    print(quad_tols)
    print(times)
    print(scores)
    
    return


def check_integrand_max_for_diff_nCells(outdir):
    n11,n10,n01,n00 = [20000,19970,30,10000]
    FP = 0.004
    FN = 0.334
    phis = np.linspace(0,2,100)
    for model, integrand, jac in [[1,_M1_logged_integrand,_M1_logged_integrand_jacobian],[2,_M2_logged_integrand,_M2_logged_integrand_jacobian],[4,_M4_logged_integrand,_M4_logged_integrand_jacobian],[5,_M5_logged_integrand,_M5_logged_integrand_jacobian]]:
        to_min = lambda x: -integrand(x[0],x[1],FP,FP,FN,FN,n11,n10,n01,n00)
        to_min_jac = lambda x: -jac(x[0],x[1],FP,FP,FN,FN,n11,n10,n01,n00)
        # min_res = optimize.minimize(to_min, [0.5,0.5], method="Nelder-Mead")
        min_res = optimize.minimize(to_min, [0.5,0.5], jac=to_min_jac, method="L-BFGS-B", bounds = ((0,1),(0,1)))
        x_min = min_res['x']

        toplot = [[to_min([i,j]) for i in phis] for j in phis]
        plt.figure()
        plt.pcolormesh(phis,phis,toplot,shading="nearest")
        plt.plot(x_min[0],x_min[1],"r.")
        plt.colorbar()
        plt.savefig(os.path.join(outdir,"model{}.png".format(model)))
    return


def main():

    sim_dat_dir = os.path.join(DATA_DIR,"simulated")
    # sim = "tree1_es3_phis1_nc300"
    # sim = "tree1_es1_phis1_nc5000"
    # sim = "tree1_es1_phis1_nc50000"
    sim = "tree5_es3_phis1_nc395"
    # sim = "tree6_es1_phis1_nc1080"
    # sim = "tree6_es1_phis1_nc2160"
    # sim = "tree6_es1_phis1_nc3240"

    sim_data_fn = os.path.join(DATA_DIR, "simulated", sim+"_data.txt")
    outdir = os.path.join(OUT_DIR, "testing_quad_ver", "w_split_MLE_scaling", sim)
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    
    # test_the_MLE_calcs(os.path.join(OUT_DIR, "testing_quad_ver", "w_split_MLE_scaling","check_returned_MLE_phis"))
    # return

    # check_out_negative_score_error(os.path.join(OUT_DIR, "testing_quad_ver", "w_split_MLE_scaling", "looking_into_neg_scores_error"))
    # return

    # check_integrand_max_for_diff_nCells(os.path.join(OUT_DIR, "testing_quad_ver", "w_split_MLE_scaling", "looking_into_failure_for_high_nCells"))
    # return

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

    # test_full_integrand_MLE_calcs(data,FP,FN,outdir)
    # return

    # compare_anc_tensor_contructors(data,FP,FN,outdir)

    compare_run_times_for_diff_tol_settings(data,FP,FN,os.path.join(outdir,"comping_min_quad_tols"))

    # start = time.time()
    # plot_resulting_anc_mats(data,FP,FN,outdir)
    # end = time.time()
    # print("Took {}sec to run".format(end-start))

    return



if __name__ == "__main__":
    main()