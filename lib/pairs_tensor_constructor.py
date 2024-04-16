import warnings
import numpy as np
import time
import multiprocessing
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize
from scipy.special import logsumexp
from numba import njit

from common import Models, NUM_MODELS, _EPSILON
from util import determine_all_mutation_pair_occurance_counts
# from pairs_tensor_util import p_data_given_model_phis_and_errors, log_p_data_given_model_phis_and_errors, p_model
from pairs_tensor_util import log_p_cluster_data_given_model_phis_and_errors, p_cluster_data_given_model_phis_and_errors, p_model

# @njit
def _2d_integrand(phi1,phi2,model,pairwise_occurances,clust1,clust2,clust_ass,fprs,ados,d_rng_i,scale):
    return p_cluster_data_given_model_phis_and_errors(model,pairwise_occurances,clust1,clust2,clust_ass,phi1,phi2,fprs,ados,d_rng_i,scale)

# @njit
def _1d_integrand(phi,model,pairwise_occurances,clust1,clust2,clust_ass,fprs,ados,d_rng_i,scale):
    return p_cluster_data_given_model_phis_and_errors(model,pairwise_occurances,clust1,clust2,clust_ass,phi,phi,fprs,ados,d_rng_i,scale)

# @njit
def _2d_tomin(x,model,pairwise_occurances,clust1,clust2,clust_ass,fprs,ados,d_rng_i):
    return -log_p_cluster_data_given_model_phis_and_errors(model,pairwise_occurances,clust1,clust2,clust_ass,x[0],x[1],fprs,ados,d_rng_i)

# @njit
def _1d_tomin(x,model,pairwise_occurances,clust1,clust2,clust_ass,fprs,ados,d_rng_i):
    return -log_p_cluster_data_given_model_phis_and_errors(model,pairwise_occurances,clust1,clust2,clust_ass,x[0],x[0],fprs,ados,d_rng_i)

def _get_model_params(model, d_rng_i):
    if model==Models.A_B:
        to_integrate = _2d_integrand
        to_min       = _2d_tomin
        x0s = [[x,y] for x in np.linspace(0.1,0.9,5) for y in np.linspace(0.1,0.9,5) if y<x] #minimization starting point
        L = lambda x: x #Integration lower bound
        U = lambda x: 1 #Integration upper bound
    elif model==Models.B_A:
        to_integrate = _2d_integrand
        to_min       = _2d_tomin
        x0s = [[x,y] for x in np.linspace(0.1,0.9,5) for y in np.linspace(0.1,0.9,5) if y>x]
        L = lambda x: 0
        U = lambda x: x
    elif model==Models.cocluster:
        to_integrate = _1d_integrand
        to_min = _1d_tomin
        x0s = [[x] for x in np.linspace(0.01,0.99,10)]
        L = lambda x: None
        U = lambda x: None
    elif model==Models.diff_branches:
        to_integrate = _2d_integrand
        to_min       = _2d_tomin
        x0s = [[x,y] for x in np.linspace(0.1,0.9,5) for y in np.linspace(0.1,0.9,5) if (x+y)<1]
        L = lambda x: 0
        U = lambda x: 1-x
    elif model==Models.garbage:
        to_integrate = _2d_integrand
        to_min       = _2d_tomin
        x0s = [[x,y] for x in np.linspace(0.1,0.9,5) for y in np.linspace(0.1,0.9,5)]
        L = lambda x: 0
        U = lambda x: 1
    return to_integrate, to_min, x0s, L, U


def _calc_nonnorm_relationship_posterior(model, pairwise_occurances, fprs, ados, clust_ass, clust1, clust2, inc_cocluster, scale_integrand, quad_tol, d_rng_i):
    scale = 0
    to_integrate, to_min, x0s, L, U = _get_model_params(model, d_rng_i)
    if scale_integrand:
        #In order to avoid underflow issues during integration, determine the maximum of the integrand and scale it by that.
        #In order for this to work, I converted the integrands to work in log space and return the non-logged, scaled score.
        #Thus, once integration is complete, log the score and add the scale back onto the log(score) to remove its influence.
        
        #I found minimization to be slow and attempt to optimize it resulted in very wrong answers. Simply doing a grid search in
        #the allowable range and using that min to the function results in great runtime savings.
        x0 = x0s[np.argmin([to_min(x, model, pairwise_occurances, clust1, clust2, clust_ass, fprs, ados, d_rng_i) for x in x0s])]
        min_res = minimize(to_min, x0, method="Nelder-Mead", bounds=((0,1),(0,1)), args = (model, pairwise_occurances, clust1, clust2, clust_ass, fprs, ados, d_rng_i))
        scale = -min_res['fun']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model==Models.cocluster:
            score = quad(to_integrate, 0, 1, args=(model, pairwise_occurances, clust1, clust2, clust_ass, fprs, ados, d_rng_i, scale),epsabs=0,epsrel=quad_tol)
        else:
            score = dblquad(to_integrate, 0, 1, L, U, args=(model, pairwise_occurances, clust1, clust2, clust_ass, fprs, ados, d_rng_i, scale),epsabs=0,epsrel=quad_tol)
    post = np.log(p_model(model, inc_cocluster)) + np.log(score[0]) + scale

    if np.isinf(scale):
        print("The scale is inf...")
        print("scale =", scale)
        print("min_f_x =", min_res['x'])
        print("score =", score[0], np.log(score[0]))
    if np.isinf(np.log(score[0])):
        print("The score is inf...")
        print("scale =", scale)
        print("min_f_x =", min_res['x'])
        print("score =", score[0], np.log(score[0]))
        
        print("min fun args:")
        print("x0 =", x0)
        print("model =", model)
        print("pairwise_occurances =", pairwise_occurances[:,:])
        print("fprs =", fprs[0], fprs[1])
        print("ados =", ados[0], ados[1])
        print("d_rng_i =", d_rng_i)

    return post


def _complete_tensor(scores):
    #For speed I only actually calculate about half of the scores because of redundancy. Here I just fill in the gaps
    #so that we have full matrices.
    nClusts = scores.shape[0]
    new_t = np.zeros(scores.shape)
    new_t[:,:,Models.A_B] = scores[:,:,Models.A_B] + scores[:,:,Models.B_A].transpose()
    new_t[:,:,Models.B_A] = scores[:,:,Models.B_A] + scores[:,:,Models.A_B].transpose()
    new_t[:,:,Models.cocluster] = scores[:,:,Models.cocluster] + scores[:,:,Models.cocluster].transpose()
    new_t[:,:,Models.diff_branches] = scores[:,:,Models.diff_branches] + scores[:,:,Models.diff_branches].transpose()
    new_t[:,:,Models.garbage] = scores[:,:,Models.garbage] + scores[:,:,Models.garbage].transpose()
    #SNVs are always coincident with themselves
    new_t[range(nClusts),range(nClusts),Models.A_B] = -np.inf
    new_t[range(nClusts),range(nClusts),Models.B_A] = -np.inf
    new_t[range(nClusts),range(nClusts),Models.cocluster] = 0
    new_t[range(nClusts),range(nClusts),Models.diff_branches] = -np.inf
    new_t[range(nClusts),range(nClusts),Models.garbage] = -np.inf
    return new_t


def _normalize_pairs_tensor(pairs_tensor, ignore_coclust=False, ignore_garbage=True):
    if not np.all(pairs_tensor<=0):
        print("min, max of pairs tensor:", np.nanmin(pairs_tensor), np.nanmax(pairs_tensor))
        print("ind of max:", np.unravel_index(np.nanargmax(pairs_tensor),pairs_tensor.shape))
        print("n_neginf:", np.sum(np.isneginf(pairs_tensor)))
        print("n_posinf:", np.sum(np.isposinf(pairs_tensor)))
        print("n_nan:",  np.sum(np.isnan(pairs_tensor)))
        print("n_0:",  np.sum(pairs_tensor==0))
        print("n_1:",  np.sum(pairs_tensor==1))
        np.save("broken_pairs_tensor", pairs_tensor, fix_imports=False)

    assert np.all(pairs_tensor<=0) #I.e., is in log space

    nClusts = pairs_tensor.shape[0]
    normed = np.copy(pairs_tensor)
    if ignore_coclust:
        normed[:,:,Models.cocluster] = -np.inf
        normed[range(nClusts),range(nClusts),Models.cocluster] = 0
    if ignore_garbage:
        normed[:,:,Models.garbage] = -np.inf
    for i in range(nClusts):
        for j in range(nClusts):
            normed[i,j,:] = normed[i,j,:] - logsumexp(normed[i,j,:])
    return normed


def construct_pairs_tensor(data, fpr, ado, d_rng_i, clst_ass=None, parallel=None, quad_tol=1e0, verbose=True, scale_integrand=None, ignore_coclust=True, ignore_garbage=True):
    
    #_calc_relationship_posterior currently assumes that fpr and ado will be passed as a vector to allow for individual error rates
    #If using global error rates and so only input scalar, convert to vector
    if parallel is None:
        parallel = multiprocessing.cpu_count()
    nSNVs, nCells = data.shape
    if np.isscalar(fpr):
        fpr = np.zeros((nSNVs,)) + fpr
    if np.isscalar(ado):
        ado = np.zeros((nSNVs,)) + ado
    assert len(fpr) == nSNVs
    assert len(ado) == nSNVs
    
    if (scale_integrand is None):
        #If we are working with more than 150 cells then there is a chance that we will experience underflow issues.
        # -->Scale the integrand during integration.
        scale_integrand = nCells > 150

    mods = []
    for m in Models:
        if m == Models.cocluster and ignore_coclust:
            continue
        if m == Models.garbage and ignore_garbage:
            continue
        mods.append(m)
    
    if (clst_ass is None):
        #Let's just say each mutation is assigned its own cluster
        clst_ass = np.arange(nSNVs)
    assert len(clst_ass) == nSNVs
    clusts = np.sort(np.unique(clst_ass))
    nClusts = len(clusts)
    
    pool = multiprocessing.Pool(parallel)
    pairwise_occurances, _ = determine_all_mutation_pair_occurance_counts(data, d_rng_i)
    args = [[model, pairwise_occurances, fpr, ado, clst_ass, clusts[c1], clusts[c2], (not ignore_coclust), scale_integrand, quad_tol, d_rng_i] for model in mods for c1 in range(nClusts-1) for c2 in range(c1+1,nClusts)]
    chunksize = int(np.ceil(nSNVs*nSNVs*len(mods)/2/parallel/8))
    results = pool.starmap(_calc_nonnorm_relationship_posterior,args,chunksize=chunksize)
    pool.close()
    pool.join()
    
    tensor = np.zeros((nClusts,nClusts,NUM_MODELS))
    k=0
    for model in mods:
        for c1 in range(nClusts-1):
            for c2 in range(c1+1,nClusts):
                tensor[c1,c2,model] = results[k]
                k+=1
    pool.terminate()
    tensor = _complete_tensor(tensor)
    # tensor = _normalize_pairs_tensor(tensor, ignore_coclust=ignore_coclust, ignore_garbage=ignore_garbage)
    return tensor
