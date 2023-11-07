import numpy as np
from numba import njit
from common import Models, DataRangeIdx, DataRange


ISCLOSE_TOLERANCE = 1e-8

@njit('f8(i1,f8,f8)', cache=True)
def p_phi_given_model(model, phi_a, phi_b): #P(phi|M)
    
    if (phi_a<0) or (phi_b<0) or (phi_b>1) or (phi_b>1):
        return 0
    
    if model == Models.A_B:
        if phi_a >= phi_b:
            return 2.0
        else:
            return 0.0
    elif model == Models.B_A:
        if phi_a <= phi_b:
            return 2.0
        else:
            return 0.0
    elif model == Models.diff_branches:
        if phi_a + phi_b <= 1:
            return 2.0
        else:
            return 0.0
    elif model == Models.cocluster:
        if np.abs(phi_a-phi_b) < ISCLOSE_TOLERANCE:
            return 1.0
        else:
            return 0.0
    elif model == Models.garbage:
        return 1.0
    
    return 0.0 #necessary for numba to not get confused

@njit('f8(i1,f8,f8)', cache=True)
def _p_model_given_phi(model, phi_a, phi_b): #P(M|Phi) - Used in error rate estimation.
    
    if (phi_a<0) or (phi_b<0) or (phi_b>1) or (phi_b>1):
        return 0
    
    if model == Models.A_B:
        if phi_a >= phi_b:
            if phi_a + phi_b <= 1:
                return 0.5
            else:
                return 1.0
        else:
            return 0.0
    elif model == Models.B_A:
        if phi_a <= phi_b:
            if phi_a + phi_b <= 1:
                return 0.5
            else:
                return 1.0
        else:
            return 0.0
    elif model == Models.diff_branches:
        if phi_a + phi_b <= 1:
            return 0.5
        else:
            return 0.0
    elif model == Models.garbage:
        return 1.0
    elif model == Models.cocluster:
        raise Exception("p(M|Phi) only accepts branched, ancestral and descendent relationships as input.\n\n I am currently only using this to estimate error rates and so only these models should be used. If something changes then make something else...")
    
    return 0.0 #necessary for numba to not get confused

@njit('f8(i8,i8,f8,f8,i8)', cache=True)
def p_data_given_truth_and_errors(d, t, fpr, ado, d_rng_i=DataRangeIdx.ref_var_nodata): #p(D_ij|t,fpr,ado)
    # d=data value
    # t=hidden true value
    # fpr=single-allele false positive rate
    # ado=allelic dropout rate
    # d_type=set of possible d values. Note that this can be either (0,1), [(0,1,3)] or (0,1,2,3)
    #   if d_type=0 then we're using possible data (0,1) and
    #       0: no variant detected
    #       1: variant detected
    #   if d_type=1 then we're using possible data (0,1,3) and
    #       0: only reference allele detected
    #       1: variant detected (ref is or is not detected)
    #       3: no reads at that locus
    #   if d_type=2 then we're using possible data (0,1,2,3) and
    #       0: only reference allele detected
    #       1: heterozygous variant detected (both ref and variant detected)
    #       2: homozygous variant detected (only variant)
    #       3: no reads at that locus
    #NOTE: I know that this way of setting d_type is annoying, but passing in an array into numba and performing checks on it make numba slow waaaaaay down. Think it has to make some functional calls to python...

    if d_rng_i==DataRangeIdx.var_notvar: #(0,1)
        if (d==1) and (t==1): #TP
            return (1-ado)**2*(1-fpr+fpr**2) + ado*(1-ado)
        elif (d==1) and (t==0): #FP
            return (1-ado)**2*(fpr**2+2*fpr*(1-fpr)) + 2*ado*(1-ado)*fpr
        elif (d==0) and (t==1): #FN
            return (1-ado)**2*fpr*(1-fpr) + ado*(1-ado) + ado**2
        elif (d==0) and (t==0): #TN
            return (1-ado)**2*(1-fpr)**2 + 2*(1-ado)*ado*(1-fpr) + ado**2
    elif d_rng_i==DataRangeIdx.ref_var_nodata: #(0,1,3)
        if d==3:
            return ado**2
        elif (d==1) and (t==1): #TP
            return (1-ado)**2*(1-fpr+fpr**2) + ado*(1-ado)
        elif (d==1) and (t==0): #FP
            return (1-ado)**2*(fpr**2+2*fpr*(1-fpr)) + 2*ado*(1-ado)*fpr #(1-ado)*fpr*(2-fpr+ado)
        elif (d==0) and (t==1): #FN
            return (1-ado)**2*fpr*(1-fpr) + ado*(1-ado)
        elif (d==0) and (t==0): #TN
            return (1-ado)**2*(1-fpr)**2 + 2*ado*(1-ado)*(1-fpr)
    elif d_rng_i==DataRangeIdx.ref_hetvar_homvar_nodata: #(0,1,2,3)
        if d==3:
            return ado**2
        elif (d==2) and (t==1):
            return (1-ado)**2*fpr*(1-fpr) + ado*(1-ado)
        elif (d==2) and (t==0):
            return (1-ado)**2*fpr**2 + 2*ado*(1-ado)*fpr
        elif (d==1) and (t==1):
            return (1-ado)**2*((1-fpr)**2 + fpr**2)
        elif (d==1) and (t==0):
            return 2*(1-ado)**2*(1-fpr)*fpr
        elif (d==0) and (t==1):
            return (1-ado)**2*fpr*(1-fpr) + ado*(1-ado)
        elif (d==0) and (t==0):
            return (1-ado)**2*(1-fpr)**2 + 2*ado*(1-ado)*(1-fpr)
    return 0.0 #necessary for numba to not get confused

@njit('f8(i1,i1,i1,f8,f8)', cache=True)
def p_trueDat_given_model_and_phis(t1,t2,model,phi1,phi2):
    if model==Models.cocluster:
        if (np.abs(phi1-phi2) > ISCLOSE_TOLERANCE) or (t1 != t2):
            to_ret = 0
        elif t1==0:
            to_ret = 1-phi1
        else:
            to_ret = phi1
    elif model==Models.A_B:
        if t1==0 and t2==0:
            to_ret = 1-phi1
        elif t1==1 and t2==0:
            to_ret = phi1-phi2
        elif t1==0 and t2==1:
            to_ret = 0
        elif t1==1 and t2==1:
            to_ret = phi2
    elif model==Models.B_A:
        if t1==0 and t2==0:
            to_ret = 1-phi2
        elif t1==1 and t2==0:
            to_ret = 0
        elif t1==0 and t2==1:
            to_ret = phi2-phi1
        elif t1==1 and t2==1:
            to_ret = phi1
    elif model==Models.diff_branches:
        if t1==0 and t2==0:
            to_ret = 1-phi1-phi2
        elif t1==1 and t2==0:
            to_ret = phi1
        elif t1==0 and t2==1:
            to_ret = phi2
        elif t1==1 and t2==1:
            to_ret = 0
    elif model==Models.garbage:
        #Not checked
        if t1==0 and t2==0:
            to_ret = (1-phi1)*(1-phi2)
        elif t1==1 and t2==0:
            to_ret = phi1*(1-phi2)
        elif t1==0 and t2==1:
            to_ret = (1-phi1)*phi2
        elif t1==1 and t2==1:
            to_ret = phi1*phi2
    if to_ret<0: #Hacky way of implementing phi constraints...
        to_ret = 0        
    return to_ret 

@njit(cache=True)
def _log_p_data_given_model_phis_and_errors(model, pairwise_occurances, fpr_a, fpr_b, ado_a, ado_b, phi_a, phi_b, d_rng_i, d_rng):
    phi_pri = p_phi_given_model(model, phi_a, phi_b)
    if phi_pri==0:
        return -np.inf
    
    log_post = np.log(phi_pri)    
    for i,d_i in enumerate(d_rng):
        for j,d_j in enumerate(d_rng):
            to_sum = 0
            #This slight speeds things up. It is also necessary for cases where
            # model = cocluster and pairwise_occurances==0, as np.log(to_sum) is 
            # likely to be set to -np.inf and 0*-inf = nan. In such a case, we just
            # want to 
            # if pairwise_occurances[i,j] == 0:
            #     continue
            for t_n in (0,1):
                for t_m in (0,1):
                    to_sum += p_data_given_truth_and_errors(d_i,t_n,fpr_a,ado_a,d_rng_i) * \
                              p_data_given_truth_and_errors(d_j,t_m,fpr_b,ado_b,d_rng_i) * \
                              p_trueDat_given_model_and_phis(t_n,t_m,model,phi_a,phi_b)

            log_post += pairwise_occurances[i,j] * np.log(to_sum)
    
    return log_post

def log_p_data_given_model_phis_and_errors(model, pairwise_occurances, fpr_a, fpr_b, ado_a, ado_b, phi_a, phi_b, d_rng_i):
    # d_set=set of possible d values. Note that this can be either (0,1), [(0,1,3)] or (0,1,2,3)
    #   if d_set=0 then we're using possible data (0,1) and
    #       0: no variant detected
    #       1: variant detected
    #   if d_set=1 then we're using possible data (0,1,3) and
    #       0: only reference allele detected
    #       1: variant detected (ref is or is not detected)
    #       3: no reads at that locus
    #   if d_set=2 then we're using possible data (0,1,2,3) and
    #       0: only reference allele detected
    #       1: heterozygous variant detected (both ref and variant detected)
    #       2: homozygous variant detected (only variant)
    #       3: no reads at that locus
    #NOTE: I know that this way of setting d_set is annoying, but passing in an array into numba and performing checks on it make numba slow waaaaaay down. Think it has to make some functional calls to python...
    d_rng = DataRange[d_rng_i]
    
    #Note: these assertions don't seem to slow down runtime at all
    assert len(pairwise_occurances.shape)==2
    assert pairwise_occurances.shape[0] == pairwise_occurances.shape[1]
    assert pairwise_occurances.shape[0] == len(d_rng)
    
    return _log_p_data_given_model_phis_and_errors(model, pairwise_occurances, fpr_a, fpr_b, ado_a, ado_b, phi_a, phi_b, d_rng_i, d_rng)

def p_data_given_model_phis_and_errors(model, pairwise_occurances, fpr_a, fpr_b, ado_a, ado_b, phi_a, phi_b, d_rng_i, scale=0):
    post = np.exp(log_p_data_given_model_phis_and_errors(model,pairwise_occurances,fpr_a,fpr_b,ado_a,ado_b,phi_a,phi_b,d_rng_i) - scale)
    return post