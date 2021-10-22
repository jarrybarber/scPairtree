#So, turns out this is real slow. Possible that there is a better way to do this. Can bring up with Philip and Quaid maybe?

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from numba import jit, njit
from theano.compile.ops import as_op
from scipy.special import logsumexp

BIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'bin'))
sys.path.append(BIN_DIR)
# from data_simulator import load_tree_parameters
from util import load_sim_data, determine_pairwise_occurance_counts
from util import DATA_DIR, OUT_DIR

@njit
def M1_LLH(n11, n10, n01, n00, alpha, beta, phi1, phi2):
    if phi1<phi2: return -np.inf
    P11 =      (1-beta)**2 * phi2  +  (1-beta)*alpha     * (phi1-phi2)  +  alpha**2        * (1-phi1)
    P10 = beta*(1-beta)    * phi2  +  (1-beta)*(1-alpha) * (phi1-phi2)  +  alpha*(1-alpha) * (1-phi1)
    P01 = beta*(1-beta)    * phi2  +  beta*alpha         * (phi1-phi2)  +  alpha*(1-alpha) * (1-phi1)
    P00 = beta**2          * phi2  +  beta*(1-alpha)     * (phi1-phi2)  +  (1-alpha)**2    * (1-phi1)
    return n11*np.log(P11) + n10*np.log(P10) + n01*np.log(P01) + n00*np.log(P00)

@njit
def M2_LLH(n11, n10, n01, n00, alpha, beta, phi1, phi2):
    if phi2<phi1: return -np.inf
    P11 =      (1-beta)**2 * phi1 + (1-beta)*alpha     * (phi2-phi1) + alpha**2        * (1-phi2)
    P10 = beta*(1-beta)    * phi1 + alpha*beta         * (phi2-phi1) + alpha*(1-alpha) * (1-phi2)
    P01 = beta*(1-beta)    * phi1 + (1-alpha)*(1-beta) * (phi2-phi1) + alpha*(1-alpha) * (1-phi2)
    P00 = beta**2          * phi1 + beta*(1-alpha)     * (phi2-phi1) + (1-alpha)**2    * (1-phi2)
    return n11*np.log(P11) + n10*np.log(P10) + n01*np.log(P01) + n00*np.log(P00)

# @njit
# def M3_LLH(n11, n10, n01, n00, alpha, beta, phi):
#     P11 =      (1-beta)**2 * phi + alpha**2        * (1-phi)
#     P10 = beta*(1-beta)    * phi + alpha*(1-alpha) * (1-phi)
#     P01 = beta*(1-beta)    * phi + alpha*(1-alpha) * (1-phi)
#     P00 = beta**2          * phi + (1-alpha)**2    * (1-phi)
#     return P11**n11 * P10**n10 * P01**n01 * P00**n00

@njit
def M4_LLH(n11, n10, n01, n00, alpha, beta, phi1, phi2):
    if phi1+phi2>1: return -np.inf
    P11 = alpha*(1-beta)     * phi1 + (1-beta)*alpha     * phi2 + alpha**2        * (1-phi1-phi2)
    P10 = (1-alpha)*(1-beta) * phi1 + alpha*beta         * phi2 + alpha*(1-alpha) * (1-phi1-phi2)
    P01 = alpha*beta         * phi1 + (1-alpha)*(1-beta) * phi2 + alpha*(1-alpha) * (1-phi1-phi2)
    P00 = beta*(1-alpha)     * phi1 + beta*(1-alpha)     * phi2 + (1-alpha)**2    * (1-phi1-phi2)
    return n11*np.log(P11) + n10*np.log(P10) + n01*np.log(P01) + n00*np.log(P00)


def lets_try_some_shit(data, outdir):

    n11,n10,n01,n00 = determine_pairwise_occurance_counts(data)
    nMuts = n11.shape[0]

    @as_op(itypes=[tt.dscalar,tt.dscalar,tt.dvector], \
            otypes=[tt.dscalar])
    def calc_full_LLH(a,b,phis):
        LLH = np.sum([logsumexp([M1_LLH(n11[i,j],n10[i,j],n01[i,j],n00[i,j],a,b,phis[i],phis[j]),\
                                M2_LLH(n11[i,j],n10[i,j],n01[i,j],n00[i,j],a,b,phis[i],phis[j]),\
                                M4_LLH(n11[i,j],n10[i,j],n01[i,j],n00[i,j],a,b,phis[i],phis[j])
                            ])\
                    for i in range(0,nMuts) for j in range(i+1,nMuts)])
        return np.array(LLH)

    with pm.Model() as model:
        phis = tt.stack([pm.Uniform("phi"+str(i),0,1) for i in range(nMuts)])
        # phis = pm.Uniform("phis",0,1,shape=nMuts) #Make this a vector
        a  = pm.Beta("FP_rate",alpha=1,beta=1)
        b  = pm.Beta("FN_rate",alpha=1,beta=1)
        # phis = tt.stack([pm.Beta("phi_"+str(i), alpha=n1[i] + b*n0[i], beta=n0[i]+a*n1[i]) for i in range(nMuts)])
        like = pm.Potential('like', calc_full_LLH(a,b,phis))
        step = pm.Metropolis()
        trace = pm.sample(20000,tune=50,chains=1,step=step)

    # trace = trace[15000:]

    phis_after_run = np.transpose(np.array([trace['phi'+str(i)] for i in range(nMuts)]))

    plt.subplot(211)
    plt.plot(trace['FP_rate'])
    plt.title('FP rate')
    plt.subplot(212)
    plt.plot(trace['FN_rate'])
    plt.title('FN rate')
    plt.savefig(os.path.join(outdir,"error_rate_traces.png"))
    plt.close()
    
    print(phis_after_run)
    print(phis_after_run.shape)

    final_a = np.mean(trace['FP_rate'])
    final_b = np.mean(trace['FN_rate'])
    final_phis = np.mean(phis_after_run,axis=0)

    print(final_a)
    print(final_b)
    print(final_phis)

    scores = np.zeros((nMuts,nMuts,3))
    for i in range(0,nMuts):
        for j in range(i+1,nMuts):
            scores[i,j,0] = M1_LLH(n11[i,j],n10[i,j],n01[i,j],n00[i,j], final_a, final_b, final_phis[i], final_phis[j])
            scores[i,j,1] = M2_LLH(n11[i,j],n10[i,j],n01[i,j],n00[i,j], final_a, final_b, final_phis[i], final_phis[j])
            scores[i,j,2] = M4_LLH(n11[i,j],n10[i,j],n01[i,j],n00[i,j], final_a, final_b, final_phis[i], final_phis[j])

    best_models = np.argmax(scores,axis=2)
    plt.figure()
    plt.imshow(best_models)
    plt.title("best_models after all that")
    plt.savefig(os.path.join(outdir,"best_models.png"))
    plt.close()

    maxScores = np.max(scores)
    minScores = np.min(scores)

    plt.subplot(2,2,1)
    plt.imshow(scores[:,:,0],vmax=maxScores)
    plt.title("A->B")
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(scores[:,:,1],vmax=maxScores)
    plt.title("B->A")
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(scores[:,:,2],vmax=maxScores)
    plt.title("Branched")
    plt.colorbar()
    plt.savefig(os.path.join(outdir,"raw_scores.png"))
    plt.close()

    plt.plot(final_phis)
    plt.savefig(os.path.join(outdir,"final_phis.png"))
    plt.close()

    return


def main():
    fn = "tree1_2_data.txt"
    outdir = os.path.join(OUT_DIR,'testing_MCMC_to_est_all_params_and_tensor')
    
    print("Loading data")
    data, _ = load_sim_data(fn)

    print("Trying some shit")
    lets_try_some_shit(data, outdir)

if __name__ == "__main__":
    main()