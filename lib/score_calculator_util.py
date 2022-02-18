import numpy as np
from numba import njit, int8, float32, float64
from common import Models


ISCLOSE_TOLERANCE = 1e-8

@njit(float32(int8, float64, float64))
def model_prior(model,phi_a,phi_b):
    
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

@njit(float64(int8, int8, float64, float64))
def p_data_given_truth_and_errors(d,t,fpr,ado): #p(D_ii|t,fpr,ado)
    #d=data,t=true,fpr=single-allele false positive rate,ado=allelic dropout rate
    assert d in (0,1,3)
    assert t in (0,1)
    if d==3:
        return ado**2 #Note: can consider incorportating copy numbers here
    if (d==1) and (t==1): #TP
        return (1-ado)**2*(1-fpr+fpr**2) + ado*(1-ado)#(1-ado**2)*(1-ado)
    if (d==1) and (t==0): #FP
        return (1-ado)*fpr*(2-fpr+ado)
    if (d==0) and (t==1): #FN
        return (1-ado)**2*fpr*(1-fpr) + ado*(1-ado)
    if (d==0) and (t==0): #TN
        return (1-ado)**2*(1-fpr)**2 + 2*(1-ado)*ado*(1-fpr)
    return 0.0 #necessary for numba to not get confused

@njit(float32(int8,int8,int8,float64,float64))
def p_trueDat_given_model_and_phis(t1,t2,model,phi1,phi2):
    if model==Models.cocluster:
        if (phi1 != phi2) | (t1 != t2):
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

def model_posterior(model,pairwise_occurances,fpr_a,fpr_b,ado_a,ado_b,phi_a,phi_b,scale=0):
    post = np.exp(log_model_posterior(model,pairwise_occurances,fpr_a,fpr_b,ado_a,ado_b,phi_a,phi_b) - scale)
    return post


@njit
def log_model_posterior(model,pairwise_occurances,fpr_a,fpr_b,ado_a,ado_b,phi_a,phi_b):
    mod_pri = model_prior(model, phi_a, phi_b)
    if mod_pri==0:
        return -np.inf

    log_post = np.log(mod_pri) + \
            np.sum(
                np.array([np.log(np.sum(np.array([np.exp(
                    np.log(p_data_given_truth_and_errors(d_i,t_n,fpr_a,ado_a))+
                    np.log(p_data_given_truth_and_errors(d_j,t_m,fpr_b,ado_b))+
                    np.log(p_trueDat_given_model_and_phis(t_n,t_m,model,phi_a,phi_b))
                    )for t_n in (0,1) for t_m in (0,1)])))*pairwise_occurances[i][j] 
                for i,d_i in enumerate((0,1,3)) for j,d_j in enumerate((0,1,3))]))
    
    return log_post

