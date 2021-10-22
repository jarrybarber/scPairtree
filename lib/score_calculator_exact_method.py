#This has been archived (May 17, 2021) because it uses the old method of expanding out the polynomial and calculating the integral.
#From here, I am going to switch to using Quadrature. This will be faster and probably just as accurate. Also don't need to make
#that assumption that FP rate << FN rate.
#(May 20, 2021) unarchived. Gotta do some testing.
#JB
import os, sys
import numpy as np
import time
from scipy.special import loggamma, logsumexp, betainc, beta

from util import log_bec, log_tec, log_qec
from multiprocessing import Pool
from numba import jit, njit


@njit
def _get_M1_inds(n1,n2,n3,n4,alpha_cutoff):
    inds = np.transpose(np.array([[i1,j1,n1-j1-i1, i2,j2,n2-j2-i2, i3,j3,n3-j3-i3, i4,j4,n4-j4-i4] \
        for i1 in range(n1+1)\
        for j1 in range(0,n1-i1+1) if j1+2*(n1-i1-j1)<=alpha_cutoff\
        for i2 in range(n2+1) \
        for j2 in range(0,n2-i2+1) if j1+2*(n1-i1-j1)+(n2-i2-j2)<=alpha_cutoff\
        for i3 in range(n3+1) \
        for j3 in range(0,n3-i3+1) if j1+2*(n1-i1-j1)+(n2-i2-j2)+j3+(n3-i3-j3)<=alpha_cutoff\
        for i4 in range(n4+1) \
        for j4 in range(0,n4-i4+1) \
        ]))
    return inds

@njit
def _get_M2_inds(n1,n2,n3,n4,alpha_cutoff):
    inds = np.transpose(np.array([[i1,j1,n1-j1-i1, i2,j2,n2-j2-i2, i3,j3,n3-j3-i3, i4,j4,n4-j4-i4] \
        for i1 in range(n1+1)\
        for j1 in range(0,n1-i1+1) if j1+2*(n1-i1-j1)<=alpha_cutoff\
        for i2 in range(n2+1) \
        for j2 in range(0,n2-i2+1) if j1+2*(n1-i1-j1)+j2+(n2-i2-j2)<=alpha_cutoff\
        for i3 in range(n3+1) \
        for j3 in range(0,n3-i3+1) if j1+2*(n1-i1-j1)+j2+(n2-i2-j2)+(n3-i3-j3)<=alpha_cutoff\
        for i4 in range(n4+1) \
        for j4 in range(0,n4-i4+1) \
        ]))
    return inds

@njit
def _get_M3_inds(n1,n2,n3,n4,alpha_cutoff):
    inds = np.transpose(np.array([[i1,n1-i1,0, i2,n2-i2,0, i3,n3-i3,0, i4,n4-i4,0] \
        for i1 in range(n1+1)    if 2*(n1-i1)<=alpha_cutoff\
        for i2 in range(n2+1) if 2*(n1-i1)+(n2-i2)<=alpha_cutoff\
        for i3 in range(n3+1) if 2*(n1-i1)+(n2-i2)+(n3-i3)<=alpha_cutoff\
        for i4 in range(n4+1) \
        ]))
    return inds

@njit
def _get_M4_inds(n1,n2,n3,n4,alpha_cutoff):

    inds = np.transpose(np.array([[i1,j1,n1-j1-i1, i2,j2,n2-j2-i2, i3,j3,n3-j3-i3, i4,j4,n4-j4-i4] \
        for i1 in range(n1+1) \
        for j1 in range(0,n1-i1+1) if (n1-i1-j1)<=alpha_cutoff \
        for i2 in range(n2+1) \
        for j2 in range(0,n2-i2+1) if (n1-i1-j1)+j2+(n2-i2-j2)<=alpha_cutoff \
        for i3 in range(n3+1)      if (n1-i1-j1)+j2+(n2-i2-j2)+i3<=alpha_cutoff \
        for j3 in range(0,n3-i3+1) if (n1-i1-j1)+j2+(n2-i2-j2)+i3+(n3-i3-j3)<=alpha_cutoff\
        for i4 in range(n4+1) \
        for j4 in range(0,n4-i4+1) \
        ]))
    return inds

@njit
def _get_M5_inds(n1,n2,n3,n4,alpha_cutoff):
    inds = np.transpose(np.array([[i1,n1+n2-i1,0, i2,n3+n4-i2,0, i3,n1+n3-i3,0, i4,n2+n4-i4,0] \
            for i1 in range(n1+n2+1) if (n1+n2-i1) <= alpha_cutoff \
            for i2 in range(n3+n4+1) \
            for i3 in range(n1+n3+1) if (n1+n2-i1) + (n1+n3-i3) <= alpha_cutoff\
            for i4 in range(n2+n4+1) \
            ]))
    return inds


def calc_ancestry_tensor_exact_method(alpha,beta,n11,n10,n01,n00,alpha_cutoff=2,alpha_max=0.015,models=[1,2,3,4,5], verbose=True):
    
    pool = Pool(len(models))
    results = {}
    if 1 in models:
        results[1] = pool.apply_async(calc_score,(1,alpha,beta,n11,n10,n01,n00,alpha_cutoff,alpha_max,verbose))
    if 2 in models:
        results[2] = pool.apply_async(calc_score,(2,alpha,beta,n11,n10,n01,n00,alpha_cutoff,alpha_max,verbose))
    if 3 in models:
        results[3] = pool.apply_async(calc_score,(3,alpha,beta,n11,n10,n01,n00,alpha_cutoff,alpha_max,verbose))
    if 4 in models:
        results[4] = pool.apply_async(calc_score,(4,alpha,beta,n11,n10,n01,n00,alpha_cutoff,alpha_max,verbose))
    if 5 in models:
        results[5] = pool.apply_async(calc_score,(5,alpha,beta,n11,n10,n01,n00,alpha_cutoff,alpha_max,verbose))
    pool.close()
    pool.join()
    scores = np.array([result.get() for _, result in results.items()])
    return scores


def non_normed_beta(a,b,upper):
    return betainc(a,b,upper)*beta(a,b)


def calc_score(model,alpha,beta,n11,n10,n01,n00, alpha_cutoff = 1, alpha_max = 0.015, verbose = True):

    if model == 1:
        # Model 1: A --> B
        get_inds_fun = _get_M1_inds
        phi_prior = np.log(2)
        phi_marginalization_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4,n1,n2,n3,n4: \
                                        loggamma(i1+i2+i3+i4+1) + loggamma(j1+j2+j3+j4+1) + loggamma(k1+k2+k3+k4+1) - loggamma(n1+n2+n3+n4+3)
        poly_expansion_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        log_tec(i1+j1+k1,i1,j1,k1) + log_tec(i2+j2+k2,i2,j2,k2) + log_tec(i3+j3+k3,i3,j3,k3) + log_tec(i4+j4+k4,i4,j4,k4)
        
        if hasattr(alpha, '__len__') and (not isinstance(alpha, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            #Complete beta integration
            # alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(2*k1+k2+k3+j1+j3+alpha[0]) + loggamma(k2+k3+2*k4+j2+j4+alpha[1]) - loggamma(j1+j2+j3+j4+2*(k1+k2+k3+k4)+alpha[0]+alpha[1]) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
            #Incomplete beta integration
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: np.log(non_normed_beta(2*k1+k2+k3+j1+j3+alpha[0], k2+k3+2*k4+j2+j4+alpha[1],alpha_max)) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
        else: #is a scalar, and so no prior
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: (j1+2*k1+k2+j3+k3)*np.log(alpha) + (j2+k2+k3+j4+2*k4)*np.log(1-alpha)
        
        if hasattr(beta, '__len__') and (not isinstance(beta, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            beta_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(i2+i3+j3+2*i4+j4+beta[0]) + loggamma(2*i1+j1+i2+j2+i3+beta[1]) - loggamma(2*(i1+i2+i3+i4)+j1+j2+j3+j4+beta[0]+beta[1]) + loggamma(beta[0]+beta[1]) - loggamma(beta[0]) - loggamma(beta[1])
        else: #is a scalar, and so no prior
            beta_contribution =  lambda beta,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4:  (i2+i3+j3+2*i4+j4)*np.log( beta) + (2*i1+j1+i2+j2+i3)*np.log(1- beta)
    elif model == 2:
        # Model 2: B --> A
        get_inds_fun = _get_M2_inds
        phi_prior = np.log(2)
        phi_marginalization_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4,n1,n2,n3,n4: \
                                        loggamma(i1+i2+i3+i4+1) + loggamma(j1+j2+j3+j4+1) + loggamma(k1+k2+k3+k4+1) - loggamma(n1+n2+n3+n4+3)
        poly_expansion_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        log_tec(i1+j1+k1,i1,j1,k1) + log_tec(i2+j2+k2,i2,j2,k2) + log_tec(i3+j3+k3,i3,j3,k3) + log_tec(i4+j4+k4,i4,j4,k4)
        
        if hasattr(alpha, '__len__') and (not isinstance(alpha, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            #Complete beta integration
            # alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(j1+2*k1+j2+k2+k3+alpha[0]) + loggamma(k2+j3+k3+j4+2*k4+alpha[1]) - loggamma(j1+j2+j3+j4+2*(k1+k2+k3+k4)+alpha[0]+alpha[1]) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
            #Incomplete beta integration
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: np.log(non_normed_beta(j1+2*k1+j2+k2+k3+alpha[0], k2+j3+k3+j4+2*k4+alpha[1],alpha_max)) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
        else: #is a scalar, and so no prior
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        (j1+2*k1+j2+k2+k3)*np.log(  alpha) + \
                                        (k2+j3+k3+j4+2*k4)*np.log(1-alpha)
        if hasattr(beta, '__len__') and (not isinstance(beta, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            beta_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(i2+j2+2*i4+j4+i3+beta[0]) + loggamma(2*i1+j1+i3+j3+i2+beta[1]) - loggamma(2*(i1+i2+i3+i4)+j1+j2+j3+j4+beta[0]+beta[1]) + loggamma(beta[0]+beta[1]) - loggamma(beta[0]) - loggamma(beta[1])
        else: #is a scalar, and so no prior
            beta_contribution =  lambda beta,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        (i2+j2+i3+2*i4+j4)*np.log(   beta) + \
                                        (2*i1+j1+i2+i3+j3)*np.log(1- beta)
    elif model == 3:
        # Model 3: AB
        get_inds_fun = _get_M3_inds
        phi_prior = 0
        phi_marginalization_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4,n1,n2,n3,n4: \
                                        loggamma(i1+i2+i3+i4+1) + loggamma(j1+j2+j3+j4+1) - loggamma(n1+n2+n3+n4+2)
        poly_expansion_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        log_bec(i1+j1,i1,j1) + log_bec(i2+j2,i2,j2) + log_bec(i3+j3,i3,j3) + log_bec(i4+j4,i4,j4)
        
        if hasattr(alpha, '__len__') and (not isinstance(alpha, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            #Complete beta integration
            # alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(2*j1+j2+j3+alpha[0]) + loggamma(j3+j2+2*j4+alpha[1]) - loggamma(2*(j1+j2+j3+j4)+alpha[0]+alpha[1]) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
            #Incomplete beta integration
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: np.log(non_normed_beta(2*j1+j2+j3+alpha[0], j3+j2+2*j4+alpha[1],alpha_max)) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
        else: #is a scalar, and so no prior
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        (2*j1+j2+j3)*np.log(  alpha) + \
                                        (j2+j3+2*j4)*np.log(1-alpha)
        if hasattr(beta, '__len__') and (not isinstance(beta, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            beta_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(i2+i3+2*i4+beta[0]) + loggamma(2*i1+i2+i3+beta[1]) - loggamma(2*(i1+i2+i3+i4)+beta[0]+beta[1]) + loggamma(beta[0]+beta[1]) - loggamma(beta[0]) - loggamma(beta[1])
        else: #is a scalar, and so no prior
            beta_contribution =  lambda beta,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        (i2+i3+2*i4)*np.log(   beta) + \
                                        (2*i1+i2+i3)*np.log(1- beta)
    elif model == 4:
        # Model 4: branching
        get_inds_fun = _get_M4_inds
        phi_prior = np.log(2)
        phi_marginalization_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4,n1,n2,n3,n4: \
                                        loggamma(i1+i2+i3+i4+1) + loggamma(j1+j2+j3+j4+1) + loggamma(k1+k2+k3+k4+1) - loggamma(n1+n2+n3+n4+3)
        poly_expansion_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        log_tec(i1+j1+k1,i1,j1,k1) + log_tec(i2+j2+k2,i2,j2,k2) + log_tec(i3+j3+k3,i3,j3,k3) + log_tec(i4+j4+k4,i4,j4,k4)
        
        if hasattr(alpha, '__len__') and (not isinstance(alpha, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            #Complete beta integration
            # alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(j1+2*k1+j2+k2+i1+i3+k3+alpha[0]) + loggamma(j3+k3+j4+2*k4+i2+k2+i4+alpha[1]) - loggamma(i1+i2+i3+i4+2*(k1+k2+k3+k4)+j1+j2+j3+j4+alpha[0]+alpha[1]) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
            #Incomplete beta integration
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: np.log(non_normed_beta(j1+2*k1+j2+k2+i1+i3+k3+alpha[0], j3+k3+j4+2*k4+i2+k2+i4+alpha[1],alpha_max)) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
        else: #is a scalar, and so no prior
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        (i1+j1+2*k1+j2+k2+i3+k3)*np.log(  alpha) + \
                                        (i2+k2+j3+k3+i4+j4+2*k4)*np.log(1-alpha)
        if hasattr(beta, '__len__') and (not isinstance(beta, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            beta_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(i3+i4+j2+j4+beta[0]) + loggamma(i1+i2+j1+j3+beta[1]) - loggamma(i1+i2+i3+i4+j1+j2+j3+j4+beta[0]+beta[1]) + loggamma(beta[0]+beta[1]) - loggamma(beta[0]) - loggamma(beta[1])
        else: #is a scalar, and so no prior
            beta_contribution =  lambda beta,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        (j2+i3+i4+j4)*np.log(   beta) + \
                                        (i1+i2+j1+j3)*np.log(1- beta)
    elif model == 5:
        # Model 5: garbage
        get_inds_fun = _get_M5_inds
        phi_prior = 0
        phi_marginalization_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4,n1,n2,n3,n4: \
                                        loggamma(i1+i2+1) + loggamma(i3+i4+1) + loggamma(j1+j2+1) + loggamma(j3+j4+1) - 2*loggamma(n1+n2+n3+n4+2)
        poly_expansion_factor = lambda i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        log_bec(n1+n2,i1,j1) + log_bec(n3+n4,i2,j2) + log_bec(n1+n3,i3,j3) + log_bec(n2+n4,i4,j4)

        if hasattr(alpha, '__len__') and (not isinstance(alpha, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            #Complete beta integration
            # alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(j1+j3+alpha[0]) + loggamma(j2+j4+alpha[1]) - loggamma(j1+j2+j3+j4+alpha[0]+alpha[1]) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
            #Incomplete beta integration
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: np.log(non_normed_beta(j1+j3+alpha[0], j2+j4+alpha[1],alpha_max)) + loggamma(alpha[0]+alpha[1]) - loggamma(alpha[0]) - loggamma(alpha[1])
        else: #is a scalar, and so no prior
            alpha_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        (j1+j3)*np.log(  alpha) + \
                                        (j2+j4)*np.log(1-alpha)
        if hasattr(beta, '__len__') and (not isinstance(beta, str)): #is a list, and so represents parameters of the beta distribution for its prior.
            beta_contribution = lambda alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: loggamma(i1+i3+beta[1]) + loggamma(i2+i4+beta[0]) - loggamma(i1+i2+i3+i4+beta[0]+beta[1]) + loggamma(beta[0]+beta[1]) - loggamma(beta[0]) - loggamma(beta[1])
        else: #is a scalar, and so no prior
            beta_contribution =  lambda beta,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4: \
                                        (i2+i4)*np.log(   beta) + \
                                        (i1+i3)*np.log(1- beta)

    nSNVs = n11.shape[0]
    
    scores = np.zeros((nSNVs,nSNVs))
    count = 0
    for s1 in range(nSNVs):
        for s2 in range(s1,nSNVs):
            if verbose:
                print("\r", 100.*count/(nSNVs*(nSNVs+1)/2), "% complete", end='   ')
            count += 1
            n1 = n11[s1,s2]
            n2 = n10[s1,s2]
            n3 = n01[s1,s2]
            n4 = n00[s1,s2]

            inds = get_inds_fun(n1,n2,n3,n4,alpha_cutoff)
            i1,j1,k1,i2,j2,k2,i3,j3,k3,i4,j4,k4 = inds
            #Calculate each term that will have to be summed together.
            tosum = phi_prior + \
                phi_marginalization_factor(i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4,n1,n2,n3,n4) + \
                poly_expansion_factor(i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4) + \
                alpha_contribution(alpha,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4) + \
                beta_contribution(beta,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4)

            #By setting b to signs, logsumexp will multiply each exped term by it's proper sign.
            scores[s1,s2] = logsumexp(tosum)
    if verbose:
        print(' ')
    return scores
