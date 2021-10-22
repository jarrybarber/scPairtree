import numpy as np
import time
from multiprocessing import Pool
from scipy.integrate import quad, dblquad
from scipy.optimize import minimize

from util import determine_pairwise_occurance_counts
from score_calculator_util import _M1_integrand, _M2_integrand, _M3_integrand, _M4_integrand, _M5_integrand
from score_calculator_util import _M1_logged_integrand, _M2_logged_integrand, _M3_logged_integrand, _M4_logged_integrand, _M5_logged_integrand



def calc_score(data,model,alphas,betas,min_tol=1,quad_tol=1e-2,verbose=True,scale_integrand=None):
    
    n11,n10,n01,n00 = determine_pairwise_occurance_counts(data)

    nSNVs = n11.shape[0]
    assert nSNVs == len(alphas)
    assert nSNVs == len(betas)

    if (scale_integrand is None):
        #If we are working with more than 200 cells then there is a chance that we will experience underflow issues.
        # -->Scale the integrand during integration.
        scale_integrand = np.any((n11+n10+n01+n00) > 200)

    if model==1:
        to_integrate = _M1_integrand
        to_min       = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M1_logged_integrand(x[0],x[1],a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = [0.5,0.5]  #minimization starting point
        L = lambda x: x #Integration lower bound
        U = lambda x: 1 #Integration upper bound
    elif model==2:
        to_integrate = _M2_integrand
        to_min       = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M2_logged_integrand(x[0],x[1],a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = [0.5,0.5]
        L = lambda x: 0
        U = lambda x: x
    elif model==3:
        to_integrate = _M3_integrand
        to_min       = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M3_logged_integrand(x,a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = 0.5
        L = lambda x: None
        U = lambda x: None
    elif model==4:
        to_integrate = _M4_integrand
        to_min       = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M4_logged_integrand(x[0],x[1],a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = [0.5,0.5]
        L = lambda x: 0
        U = lambda x: 1-x
    elif model==5:
        to_integrate = _M5_integrand
        to_min       = lambda x,a1,a2,b1,b2,n11,n10,n01,n00: -_M5_logged_integrand(x[0],x[1],a1,a2,b1,b2,n11,n10,n01,n00)
        x0 = [0.5,0.5]
        L = lambda x: 0
        U = lambda x: 1
    

    scores = np.zeros((nSNVs,nSNVs))
    count = 0
    scale = 0
    min_rts = []
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
                min_res = minimize(to_min, x0, method="Nelder-Mead", args = (alphas[s1], alphas[s2], betas[s1], betas[s2], n11[s1,s2], n10[s1,s2], n01[s1,s2], n00[s1,s2]),options={"xatol":1e-2,"fatol":min_tol*(n11[s1,s2] + n10[s1,s2] + n01[s1,s2] + n00[s1,s2])})
                scale = -min_res['fun']
                end = time.time()
                min_rts.append(end-start)

            if model==3:
                score = quad(to_integrate, 0, 1, args=(alphas[s1], alphas[s2], betas[s1], betas[s2], n11[s1,s2], n10[s1,s2], n01[s1,s2], n00[s1,s2], scale),epsabs=0,epsrel=quad_tol)
            else:
                score = dblquad(to_integrate, 0, 1, L, U, args=(alphas[s1], alphas[s2], betas[s1], betas[s2], n11[s1,s2], n10[s1,s2], n01[s1,s2], n00[s1,s2], scale),epsabs=0,epsrel=quad_tol)
            scores[s1,s2] = np.log(score[0]) + scale
    print("Avg min time =", np.mean(min_rts))
    if verbose:
        print(' ')
    return scores



def calc_ancestry_tensor(alpha, beta, n11, n10, n01, n00, models=[1,2,3,4,5], min_tol=1e-2, quad_tol=1e-2, verbose=True, scale_integrand=None):
    
    #calc_score currently assumes that alpha and beta will be passed as a vector to allow for individual error rates
    #If using global error rates and so only input scalar, convert to vector
    nSNVs = n11.shape[0]
    if np.isscalar(alpha):
        alpha = np.zeros((nSNVs,)) + alpha
    if np.isscalar(beta):
        beta = np.zeros((nSNVs,)) + beta
    assert len(alpha) == nSNVs
    assert len(beta) == nSNVs

    pool = Pool(len(models))
    results = {}
    if 1 in models:
        results[1] = pool.apply_async(calc_score,(1,alpha,beta,n11,n10,n01,n00,min_tol,quad_tol,verbose))
    if 2 in models:
        results[2] = pool.apply_async(calc_score,(2,alpha,beta,n11,n10,n01,n00,min_tol,quad_tol,verbose))
    if 3 in models:
        results[3] = pool.apply_async(calc_score,(3,alpha,beta,n11,n10,n01,n00,min_tol,quad_tol,verbose))
    if 4 in models:
        results[4] = pool.apply_async(calc_score,(4,alpha,beta,n11,n10,n01,n00,min_tol,quad_tol,verbose))
    if 5 in models:
        results[5] = pool.apply_async(calc_score,(5,alpha,beta,n11,n10,n01,n00,min_tol,quad_tol,verbose))
    pool.close()
    pool.join()
    scores = np.array([result.get() for _, result in results.items()])
    return scores