#Calculates the error rates for each mutation individually.
# --> This version works with the score calculator that makes use of 3s (dropouts)
# --> This version takes the false positive rate to be the same across mutations, and the false negative rate to be individualistic
# --> 

import numpy as np
from scipy.optimize import minimize

from pairs_tensor_util import _log_p_data_given_model_phis_and_errors, p_phi_given_model, p_model_given_phi
from util import  determine_all_mutation_pair_occurance_counts
from common import Models, DataRangeIdx, DataRange
from common import _EPSILON
from numba import njit

# def _old_non_njit_calc_to_min_val(alphas,betas,phis,pairwise_occurances,d_rng_i):
#     n_mut = len(alphas)
#     assert n_mut==len(betas)
#     assert n_mut==len(phis)

#     AB_scores     = np.array([[    np.log(p_model_given_phi(Models.A_B,phis[i],phis[j]))
#                                  - np.log(p_phi_given_model(Models.A_B,phis[i],phis[j]))
#                                  + _log_p_data_given_model_phis_and_errors(Models.A_B,pairwise_occurances[:,:,i,j],alphas[i],alphas[j],betas[i],betas[j],phis[i],phis[j],d_rng_i) 
#                                 for j in range(n_mut)]
#                             for i in range(n_mut)])
#     BA_scores     = np.transpose(AB_scores)
#     branch_scores = np.array([[   np.log(p_model_given_phi(Models.diff_branches,phis[i],phis[j]))
#                                 - np.log(p_phi_given_model(Models.diff_branches,phis[i],phis[j]))
#                                 + _log_p_data_given_model_phis_and_errors(Models.diff_branches,pairwise_occurances[:,:,i,j],alphas[i],alphas[j],betas[i],betas[j],phis[i],phis[j],d_rng_i)
#                                 if i>=j else 0 
#                                for j in range(n_mut)] 
#                               for i in range(n_mut)])
#     branch_scores = branch_scores + np.transpose(branch_scores)

#     # incorporate phi constraints
#     phi_mat = np.vstack(np.array([phis]*n_mut))
#     AB_scores[phi_mat>phi_mat.T] = -np.inf
#     BA_scores[phi_mat<phi_mat.T] = -np.inf
#     branch_scores[(phi_mat+phi_mat.T)>1] = -np.inf

#     # Calc the score
#     non_garb_scores= np.max(np.stack(np.array([AB_scores,BA_scores,branch_scores])),axis=0)
#     non_garb_scores[range(n_mut),range(n_mut)] = 0 #Set diagonal values to 0 as those are uninformative
#     non_garb_score = np.sum(non_garb_scores)
#     return -(non_garb_score)

@njit()
def _calc_to_min_val(alphas,betas,phis,pairwise_occurances,d_rng_i):
    n_mut = len(alphas)
    assert n_mut==len(betas)
    assert n_mut==len(phis)
    scores = np.zeros((3,n_mut,n_mut))
    for i in range(n_mut):
        for j in range(n_mut):
            if i==j:
                continue

            if phis[j]>phis[i]:
                AB_score = -np.inf
            else:
                AB_score = np.log(p_model_given_phi(Models.A_B,phis[i],phis[j])) \
                         - np.log(p_phi_given_model(Models.A_B,phis[i],phis[j])) \
                         + _log_p_data_given_model_phis_and_errors(Models.A_B,pairwise_occurances[:,:,i,j],alphas[i],alphas[j],betas[i],betas[j],phis[i],phis[j],d_rng_i)
            scores[0,i,j] = AB_score
            scores[1,j,i] = AB_score

            if j>i:
                continue
            if phis[i] + phis[j] > 1:
                branch_score = -np.inf
            else:
                branch_score =  np.log(p_model_given_phi(Models.diff_branches,phis[i],phis[j])) \
                              - np.log(p_phi_given_model(Models.diff_branches,phis[i],phis[j])) \
                              + _log_p_data_given_model_phis_and_errors(Models.diff_branches,pairwise_occurances[:,:,i,j],alphas[i],alphas[j],betas[i],betas[j],phis[i],phis[j],d_rng_i)
            scores[2,i,j] = branch_score
            scores[2,j,i] = branch_score

    non_garb_scores = np.zeros((n_mut,n_mut))
    for i in range(n_mut):
        for j in range(n_mut):
            non_garb_scores[i,j] = np.max(scores[:,i,j])
    non_garb_score = np.sum(non_garb_scores)

    return -(non_garb_score)


def estimate_error_rates(data, d_rng_i=DataRangeIdx.ref_var_nodata, variable_ado=True):

    pairwise_occurances, _ = determine_all_mutation_pair_occurance_counts(data, d_rng_i)
    n_mut = data.shape[0]

    if variable_ado:
        to_min = lambda x: _calc_to_min_val(np.array([x[0]]*n_mut),x[1:n_mut+1],x[n_mut+1:],pairwise_occurances,d_rng_i)
        bounds=[(0.000001,0.1)] + [(0.001,0.9)]*n_mut + [(0.0,1.0)]*n_mut
        beta0s  = np.random.beta(30,200,n_mut)
    else:
        to_min = lambda x: _calc_to_min_val(np.array([x[0]]*n_mut), np.array([x[1]]*n_mut), x[2:], pairwise_occurances,d_rng_i)
        bounds=[(0.000001,0.1)] + [(0.001,0.8)] + [(0.0,1.0)]*n_mut
        beta0s  = np.random.beta(30,200,1) 

    alpha0s = np.random.beta(11,1000,1)
    phi0s   = np.random.beta(np.sum(data==1,axis=1)+np.sum(data==2,axis=1)+_EPSILON, np.sum(data==0,axis=1)+_EPSILON)
    x0 = np.hstack([alpha0s,beta0s,phi0s])
    print(x0)

    res = minimize(to_min,
                x0=x0,
                method="Powell",
                bounds=bounds,
                options={"disp":True, "maxiter": 1000*len(x0), 'xtol': 1e-6, 'ftol':1e-6},
                )
    print(" ")
    print(res)
    est_errs = res.x
    est_FPRs = np.array([est_errs[0]]*n_mut)
    if variable_ado:
        est_FNRs = est_errs[1:n_mut+1]
        est_phis = est_errs[n_mut+1:]
    else:
        est_FNRs = np.array([est_errs[1]]*n_mut)
        est_phis = est_errs[2:]
    
    return (est_FPRs,est_FNRs,est_phis), x0