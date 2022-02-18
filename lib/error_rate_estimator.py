#Calculates the error rates for each mutation individually.
# --> This version works with the score calculator that makes use of 3s (dropouts)
# --> This version takes the false positive rate to be the same across mutations, and the false negative rate to be individualistic
# --> 

import numpy as np
from scipy.optimize import minimize

from score_calculator_util import log_model_posterior
from util import  determine_all_pairwise_occurance_counts
from common import Models
from common import _EPSILON


def _calc_to_min_val(alphas,betas,phis,pairwise_occurances):
    n_mut = len(alphas)
    assert n_mut==len(betas)
    assert n_mut==len(phis)
    AB_scores     = np.array([[log_model_posterior(Models.A_B,pairwise_occurances[:,:,i,j],alphas[i],alphas[j],betas[i],betas[j],phis[i],phis[j]) for j in range(n_mut)] for i in range(n_mut)])
    BA_scores     = np.transpose(AB_scores)
    branch_scores = np.array([[log_model_posterior(Models.diff_branches,pairwise_occurances[:,:,i,j],alphas[i],alphas[j],betas[i],betas[j],phis[i],phis[j]) if i>=j else 0 for j in range(n_mut)] for i in range(n_mut)])
    branch_scores = branch_scores + np.transpose(branch_scores)
    
    #incorporate phi constraints
    phi_mat = np.vstack([phis]*n_mut)
    AB_scores[phi_mat>phi_mat.T] = -np.inf
    BA_scores[phi_mat<phi_mat.T] = -np.inf
    branch_scores[(phi_mat+phi_mat.T)>1] = -np.inf
    
    #Calc the score
    non_garb_scores= np.max(np.stack([AB_scores,BA_scores,branch_scores]),axis=0)
    non_garb_scores[range(n_mut),range(n_mut)] = 0 #Set diagonal values to 0 as those are uninformative
    non_garb_score = np.sum(non_garb_scores)
    return -(non_garb_score)


def estimate_error_rates(data):

    pairwise_occurances, _ = determine_all_pairwise_occurance_counts(data)
    n_mut = data.shape[0]

    to_min = lambda x: _calc_to_min_val([x[0]]*n_mut,x[1:n_mut+1],x[n_mut+1:],pairwise_occurances)

    alpha0s = np.random.beta(11,1000,1)
    beta0s  = np.random.beta(30,200,n_mut)
    phi0s   = np.random.beta(np.sum(data==1,axis=1)+_EPSILON,np.sum(data==0,axis=1)+_EPSILON)
    x0 = np.hstack([alpha0s,beta0s,phi0s])

    res = minimize(to_min,
                x0=x0,
                method="Powell",
                bounds=[(0.000001,0.1)] + [(0.001,0.5)]*n_mut + [(0.0,1.0)]*n_mut,
                options={"disp":True, "maxiter": 1000*len(x0), 'xtol': 1e-6, 'ftol':1e-6},
                )
    est_errs = res.x
    est_FPRs = est_errs[0]
    est_FNRs = est_errs[1:n_mut+1]
    est_phis = est_errs[n_mut+1:]
    return (np.array([est_FPRs]*n_mut),est_FNRs,est_phis), x0