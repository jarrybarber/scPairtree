import numpy as np
import time
from multiprocessing import Pool
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize

from common import Models, NUM_MODELS, DataRange, DataRangeIdx
from util import determine_all_pairwise_occurance_counts
from pairs_tensor_util import model_posterior, log_model_posterior



def construct_model_posterior_matrix(data,model,alphas,betas,d_rng_i,quad_tol=1e0,verbose=True,scale_integrand=None):
    
    pairwise_occurances, _ = determine_all_pairwise_occurance_counts(data, d_rng_i)


    nSNVs, nCells = data.shape
    assert nSNVs == len(alphas)
    assert nSNVs == len(betas)

    if (scale_integrand is None):
        #If we are working with more than 150 cells then there is a chance that we will experience underflow issues.
        # -->Scale the integrand during integration.
        scale_integrand = nCells > 150

    
    if model==Models.A_B:
        to_integrate = lambda phi1,phi2,fpr1,fpr2,ado1,ado2,pairwise_occurances,scale: model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,phi1,phi2,d_rng_i,scale)
        to_min       = lambda x,fpr1,fpr2,ado1,ado2,pairwise_occurances: -log_model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,x[0],x[1],d_rng_i)
        x0s = [[x,y] for x in np.linspace(0.01,0.99,5) for y in np.linspace(0.01,0.99,5) if y<=x] #minimization starting point
        L = lambda x: x #Integration lower bound
        U = lambda x: 1 #Integration upper bound
    elif model==Models.B_A:
        to_integrate = lambda phi1,phi2,fpr1,fpr2,ado1,ado2,pairwise_occurances,scale: model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,phi1,phi2,d_rng_i,scale)
        to_min       = lambda x,fpr1,fpr2,ado1,ado2,pairwise_occurances: -log_model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,x[0],x[1],d_rng_i)
        x0s = [[x,y] for x in np.linspace(0.01,0.99,5) for y in np.linspace(0.01,0.99,5) if y>=x]
        L = lambda x: 0
        U = lambda x: x
    elif model==Models.cocluster:
        to_integrate = lambda phi,fpr1,fpr2,ado1,ado2,pairwise_occurances,scale: model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,phi,phi,d_rng_i,scale)
        def to_min(x,fpr1,fpr2,ado1,ado2,pairwise_occurances):
            return -log_model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,x[0],x[0],d_rng_i)
        x0s = [[x] for x in np.linspace(0.01,0.99,10)]
        L = lambda x: None
        U = lambda x: None
    elif model==Models.diff_branches:
        to_integrate = lambda phi1,phi2,fpr1,fpr2,ado1,ado2,pairwise_occurances,scale: model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,phi1,phi2,d_rng_i,scale)
        to_min       = lambda x,fpr1,fpr2,ado1,ado2,pairwise_occurances: -log_model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,x[0],x[1],d_rng_i)
        x0s = [[x,y] for x in np.linspace(0.01,0.99,5) for y in np.linspace(0.01,0.99,5) if (x+y)<=1]
        L = lambda x: 0
        U = lambda x: 1-x
    elif model==Models.garbage:
        to_integrate = lambda phi1,phi2,fpr1,fpr2,ado1,ado2,pairwise_occurances,scale: model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,phi1,phi2,d_rng_i,scale)
        to_min       = lambda x,fpr1,fpr2,ado1,ado2,pairwise_occurances: -log_model_posterior(model,pairwise_occurances,fpr1,fpr2,ado1,ado2,x[0],x[1],d_rng_i)
        x0s = [[x,y] for x in np.linspace(0.01,0.99,5) for y in np.linspace(0.01,0.99,5)]
        L = lambda x: 0
        U = lambda x: 1
    

    scores = np.zeros((nSNVs,nSNVs))
    count = 0
    scale = 0
    min_rts = []
    quad_rts = []
    for s1 in range(nSNVs):
        for s2 in range(s1+1,nSNVs):
            if verbose:
                print("\r", 100.*count/(nSNVs*(nSNVs+1)/2), "% complete", end='   ')
            count += 1
            
            if scale_integrand:
                #In order to avoid underflow issues during integration, determine the maximum of the integrand and scale it by that.
                #In order for this to work, I converted the integrands to work in log space and return the non-logged, scaled score.
                #Thus, once integration is complete, log the score and add the scale back onto the log(score) to remove it's influence.
                start = time.time()
                #I found minimization to be slow and attempt to optimize it resulted in very wrong answers. Simply doing a grid search in
                #the allowable range and using that min to the function results in great runtime savings.
                x0 = x0s[np.argmin([to_min(x,alphas[s1], alphas[s2], betas[s1], betas[s2], pairwise_occurances[:,:,s1,s2]) for x in x0s])]
                min_res = minimize(to_min, x0, method="Nelder-Mead", bounds=((0,1),(0,1)), args = (alphas[s1], alphas[s2], betas[s1], betas[s2], pairwise_occurances[:,:,s1,s2]))
                # print(min_res)
                scale = -min_res['fun']
                end = time.time()
                min_rts.append(end-start)
            start = time.time()
            if model==Models.cocluster:
                score = quad(to_integrate, 0, 1, args=(alphas[s1], alphas[s2], betas[s1], betas[s2], pairwise_occurances[:,:,s1,s2], scale),epsabs=0,epsrel=quad_tol)
            else:
                score = dblquad(to_integrate, 0, 1, L, U, args=(alphas[s1], alphas[s2], betas[s1], betas[s2], pairwise_occurances[:,:,s1,s2],scale),epsabs=0,epsrel=quad_tol)
            end = time.time()
            quad_rts.append(end-start)
            scores[s1,s2] = np.log(score[0]) + scale
    if verbose:
        if scale_integrand:
            print("Avg min time =", np.mean(min_rts))  
            print("Avg quad time =", np.mean(quad_rts))  
        print(' ')
    return scores



def construct_pairs_tensor(data, alpha, beta, d_rng_i, quad_tol=1e0, verbose=True, scale_integrand=None):
    
    #construct_model_posterior_matrix currently assumes that alpha and beta will be passed as a vector to allow for individual error rates
    #If using global error rates and so only input scalar, convert to vector
    nSNVs = data.shape[0]
    if np.isscalar(alpha):
        alpha = np.zeros((nSNVs,)) + alpha
    if np.isscalar(beta):
        beta = np.zeros((nSNVs,)) + beta
    assert len(alpha) == nSNVs
    assert len(beta) == nSNVs
    
    pool = Pool(NUM_MODELS)
    results = {}
    
    results[Models.A_B] = pool.apply_async(construct_model_posterior_matrix, args=(np.copy(data), Models.A_B, alpha, beta, d_rng_i, quad_tol, verbose, scale_integrand))
    results[Models.B_A] = pool.apply_async(construct_model_posterior_matrix, args=(np.copy(data), Models.B_A, alpha, beta, d_rng_i, quad_tol, verbose, scale_integrand))
    results[Models.cocluster] = pool.apply_async(construct_model_posterior_matrix, args=(np.copy(data), Models.cocluster, alpha, beta, d_rng_i, quad_tol, verbose, scale_integrand))
    results[Models.diff_branches] = pool.apply_async(construct_model_posterior_matrix, args=(np.copy(data), Models.diff_branches, alpha, beta, d_rng_i, quad_tol, verbose, scale_integrand))
    results[Models.garbage] = pool.apply_async(construct_model_posterior_matrix, args=(np.copy(data), Models.garbage, alpha, beta, d_rng_i, quad_tol, verbose, scale_integrand))
    pool.close()
    pool.join()

    scores = np.zeros((nSNVs,nSNVs,NUM_MODELS))
    for k in results:
        scores[:,:,k] = results[k].get()
    return scores


def complete_tensor(scores):
    #For speed I only actually calculate about half of the scores because of redundancy. Here I just fill in the gaps
    #so that we have full matrices.
    nSNVs = scores.shape[0]
    new_t = np.zeros(scores.shape)
    new_t[:,:,Models.A_B] = scores[:,:,Models.A_B] + scores[:,:,Models.B_A].transpose()
    new_t[:,:,Models.B_A] = scores[:,:,Models.B_A] + scores[:,:,Models.A_B].transpose()
    new_t[:,:,Models.cocluster] = scores[:,:,Models.cocluster] + scores[:,:,Models.cocluster].transpose()
    new_t[:,:,Models.diff_branches] = scores[:,:,Models.diff_branches] + scores[:,:,Models.diff_branches].transpose()
    new_t[:,:,Models.garbage] = scores[:,:,Models.garbage] + scores[:,:,Models.garbage].transpose()
    #Should always be confident of snvs being coclustered with itself
    new_t[range(nSNVs),range(nSNVs),Models.A_B] = -np.inf
    new_t[range(nSNVs),range(nSNVs),Models.B_A] = -np.inf
    new_t[range(nSNVs),range(nSNVs),Models.cocluster] = 0
    new_t[range(nSNVs),range(nSNVs),Models.diff_branches] = -np.inf
    new_t[range(nSNVs),range(nSNVs),Models.garbage] = -np.inf
    return new_t

