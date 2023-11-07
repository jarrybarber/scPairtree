import numpy as np

from pairs_tensor_util import _log_p_data_given_model_phis_and_errors, _p_model_given_phi
from util import  determine_all_pairwise_occurance_counts
from common import Models, DataRangeIdx, DataRange
import time
# from scipy.special import logsumexp
from util import numba_logsumexp
from scipy.integrate import quad
from numba import njit
import matplotlib.pyplot as plt


@njit(cache=True)
def _cluster_assignment_prior(new_clust_assignment, cluster_assignments, n_mut, dirichlet_alpha):
    n_mut_in_clust = np.sum(cluster_assignments==new_clust_assignment)
    if n_mut_in_clust > 0:
        return n_mut_in_clust / (dirichlet_alpha+n_mut-1)
    elif n_mut_in_clust == 0:
        return dirichlet_alpha / (dirichlet_alpha+n_mut-1)
    return -1.0

@njit(cache=True)
def _p_model_given_clust_ass_and_clust_phis(model, clust_ass1, clust_ass2, phi1, phi2):
    if (model == Models.cocluster):
        if (clust_ass1 == clust_ass2):
            return 1.0
        else:
            return 0.0
    else:
        if (clust_ass1 == clust_ass2):
            return 0.0
        else:
            return _p_model_given_phi(model, phi1, phi2)
    return 0.0 #Required for Numba to work

@njit(cache=True)
def _p_data_given_theta_phi_and_pi(n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng):
    models_to_sum_over = [Models.A_B, Models.B_A, Models.cocluster, Models.diff_branches]
    llh = 0
    for a in range(n_mut):
        fpr_a = fprs[a]
        adr_a = adrs[a]
        clst_ass_a = clst_ass[a]
        phi_a = phis[clst_ass_a]
        for b in range(a+1,n_mut):
            fpr_b = fprs[b]
            adr_b = adrs[b]
            clst_ass_b = clst_ass[b]
            phi_b = phis[clst_ass_b]
            tosum = np.zeros(len(models_to_sum_over))
            for mo, model in enumerate(models_to_sum_over):
                tosum[mo] = _log_p_data_given_model_phis_and_errors(model,pairwise_occurances[:,:,a,b],fpr_a,fpr_b,adr_a,adr_b,phi_a,phi_b,d_rng_i,d_rng) \
                            + np.log(_p_model_given_clust_ass_and_clust_phis(model, clst_ass_a, clst_ass_b, phi_a, phi_b))
            llh = llh + numba_logsumexp(tosum)
    return llh

@njit(cache=True)
def _mut_i_LLH_fraction(fpr_i, adr_i, phi_i, clst_ass_i, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng):
    models_to_sum_over = [Models.A_B, Models.B_A, Models.cocluster, Models.diff_branches]
    llh = 0
    for m in range(n_mut):
        if m==i:
            continue
        fpr_m = fprs[m]
        adr_m = adrs[m]
        clst_ass_m = clst_ass[m]
        phi_m = phis[clst_ass_m]
        tosum = np.zeros(len(models_to_sum_over))
        for mo, model in enumerate(models_to_sum_over):
            tosum[mo] = _log_p_data_given_model_phis_and_errors(model,pairwise_occurances[:,:,i,m],fpr_i,fpr_m,adr_i,adr_m,phi_i,phi_m,d_rng_i,d_rng) \
                        + np.log(_p_model_given_clust_ass_and_clust_phis(model, clst_ass_i, clst_ass_m, phi_i, phi_m))
        llh = llh + numba_logsumexp(tosum)
    return llh

@njit(cache=True)
def _mut_i_LH_fraction(fpr_i, adr_i, phi_i, clst_ass_i, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng, scale=0):
    return np.exp(_mut_i_LLH_fraction(fpr_i, adr_i, phi_i, clst_ass_i, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng) - scale)

@njit(cache=True)
def _phi_i_LLH_fraction(phi_i, muts_in_clust_i, clst_i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng):
    llh = 0
    models_to_sum_over = [Models.A_B, Models.B_A, Models.cocluster, Models.diff_branches]
    for a in range(n_mut):
        fpr_a = fprs[a]
        adr_a = adrs[a]
        clst_ass_a = clst_ass[a]
        if a in muts_in_clust_i:
            phi_a = phi_i
        else:
            phi_a = phis[clst_ass_a]
        for b in muts_in_clust_i:
            if a==b:
                continue
            fpr_b = fprs[b]
            adr_b = adrs[b]
            tosum = np.zeros(len(models_to_sum_over))
            # print("phi_i", phi_i)
            for mo, model in enumerate(models_to_sum_over):
                tosum[mo] = _log_p_data_given_model_phis_and_errors(model,pairwise_occurances[:,:,a,b],fpr_a,fpr_b,adr_a,adr_b,phi_a,phi_i,d_rng_i,d_rng) \
                            + np.log(_p_model_given_clust_ass_and_clust_phis(model, clst_ass_a, clst_i, phi_a, phi_i))
            llh = llh + numba_logsumexp(tosum)
            if np.isnan(llh):
                raise Exception("The llh of phi was calculated to be nan. This should not happen! Often occurs because accidentally try to calculate log(0) or 0*inf.")
    return llh

@njit(cache=True)
def _phi_i_LH_fraction(phi_i, muts_in_clust_i, clst_ass_i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng, scale=0):
    return np.exp(_phi_i_LLH_fraction(phi_i, muts_in_clust_i, clst_ass_i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng) - scale)


def _monte_carlo_sample_given_continuous_pdf(pdf, max_pdf, lower_bound, upper_bound):
    assert upper_bound > lower_bound
    samp_x = np.random.rand()*(upper_bound-lower_bound) + lower_bound
    samp_y = np.random.rand()*max_pdf
    samp_f = pdf(samp_x)
    i = 0
    while samp_y > samp_f:
        samp_x = np.random.rand()*(upper_bound-lower_bound) + lower_bound
        samp_y = np.random.rand()*max_pdf
        samp_f = pdf(samp_x)
        i += 1
        if i > 1000:
            raise Exception("Number of attempts to sample from pdf exceeded maximum allowed. \n Ensure that the pdf is well defined and that the max value input isn't too much larger than the true max(pdf).")
    return samp_x


def _find_max_and_bounds_of_logfun(fun, lb, ub, max_threshold = -1., non_zero_threshold=-30., n_init_samples = 10, max_iterations=50):
    #This is my attempt at maximizing quickly, while also determining good bounds for integration and monte carlo sampling
    # Basically, will sample evenly and find the current max.
    # Then will check if the points to the left and right of max evaluate to values close enough to the current max.
    # Once max is found, looks among previous samples for those that are less than the max minus the non_zero threshold
    # Those samples are essentiall zero (in non-log space) and so we ignore them for integration and sampling through setting bounds
    assert ub>lb
    assert n_init_samples > 2
    assert max_threshold < 0
    assert non_zero_threshold < 0
    n_samples = n_init_samples
    x, dx = np.linspace(lb, ub, num=n_samples, retstep=True)
    samples = np.array([fun(i) for i in x])
    current_max_i = np.argmax(samples)
    current_max   = samples[current_max_i]
    i = 0
    while   (current_max_i not in [0, n_samples-1] \
                and samples[current_max_i-1] < current_max+max_threshold \
                and samples[current_max_i+1] < current_max+max_threshold) \
            or (current_max_i == 0 and samples[current_max_i+1] < current_max+max_threshold) \
            or (current_max_i == n_samples-1 and samples[current_max_i-1] < current_max+max_threshold):
        dx = dx/2
        new_x = x[current_max_i] + dx
        if new_x < ub:
            x = np.insert(x,current_max_i+1, new_x)
            new_sample = fun(new_x)
            samples = np.insert(samples,current_max_i+1,new_sample)
            n_samples += 1
        new_x = x[current_max_i] - dx
        if new_x > lb:
            x = np.insert(x,current_max_i, new_x)
            new_sample = fun(new_x)
            samples = np.insert(samples,current_max_i,new_sample)
            n_samples += 1
        current_max_i = np.argmax(samples)
        current_max   = samples[current_max_i]
        i += 1
        if i>max_iterations:
            print("WARNING: Reached the maximum allowed number of iterations. Ending maximization early.")
            break

    non_zero_indicies = np.argwhere(samples > current_max+non_zero_threshold).flatten()
    lower_bound_i = np.max([0, non_zero_indicies[0]-1])
    upper_bound_i = np.min([n_samples-1,non_zero_indicies[-1]+1])
    lower_bound = x[lower_bound_i]
    upper_bound = x[upper_bound_i]

    return current_max, x[current_max_i], lower_bound, upper_bound


def _update_error_rates(i, pairwise_occurances, n_mut, fprs, adrs, phis, clst_ass, d_rng_i, d_rng):
    #If I implement, this will need touching up.
    adr_i = adrs[i]
    clst_ass_i = clst_ass[i]
    phi_i = phis[clst_ass_i]

    #update fpr_i
    #We find the maximum of the LLH so that we can scale the LH to allow for quadrature without numerical underflow
    to_min = lambda x: -_mut_i_LLH_fraction(x, adr_i, phi_i, clst_ass_i, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng)
    # llh_scale = minimize(to_min, x0=[0.05], method="Nelder-Mead", bounds=((0,1),))
    llh_scale = np.min([to_min(i) for i in np.arange(0,1,1/100)])
    fpr_i_lh = lambda x: _mut_i_LH_fraction(x, adr_i, phi_i, clst_ass_i, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng, llh_scale)
    norm_val = quad(fpr_i_lh,0,1)[0]*np.exp(llh_scale)
    fpr_post = lambda x: _mut_i_LH_fraction(x, adr_i, phi_i, clst_ass_i, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng, norm_val)
    new_fpr = _monte_carlo_sample_given_continuous_pdf(fpr_post, norm_val, 0, 1)
    fprs[i] = new_fpr
    
    #update adr_i
    to_min = lambda x: -_mut_i_LLH_fraction(new_fpr, x, phi_i, clst_ass_i, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng)
    # llh_scale = minimize(to_min, x0=0.3, method="Nelder-Mead", bounds=((0,1),))
    llh_scale = np.min([to_min(i) for i in np.arange(0,1,1/100)])
    adr_i_lh = lambda x: _mut_i_LH_fraction(new_fpr, x, phi_i, clst_ass_i, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng, llh_scale)
    norm_val = quad(adr_i_lh,0,1)[0]*np.exp(llh_scale)
    adr_post = lambda x: _mut_i_LH_fraction(new_fpr, x, phi_i, clst_ass_i, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng, norm_val)
    new_adr = _monte_carlo_sample_given_continuous_pdf(adr_post, norm_val, 0, 1)
    adrs[i] = new_adr

    return


def _update_clst_ass(i, pairwise_occurances, n_mut, n_clust, fprs, adrs, phis, clst_ass, d_rng_i, d_rng, dirichlet_alpha):
    
    fpr_i = fprs[i]
    adr_i = adrs[i]
    old_clst_ass_i = clst_ass[i]
    
    n_mut_in_clust = np.sum(clst_ass==clst_ass[i])
    if n_mut_in_clust==1:
        n_clust_to_consider = n_clust
        this_phis = np.copy(phis)
    elif n_mut_in_clust > 1:
        n_clust_to_consider = n_clust + 1
        phi_new_cluster = np.random.rand() #May want to change this in the future to be based on the number of times this mutation is observed
        this_phis = np.append(phis, phi_new_cluster) #Note: new cluster phi will be ignored if mut_i is currently the sole member of a cluster
    else:
        print(n_mut_in_clust)
        raise Exception("The number of mutations in a cluster is less than 1. Something has gone wrong.")

    clst_ass_i_llh = lambda x: _mut_i_LLH_fraction(fpr_i, adr_i, this_phis[x], x, i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng)
    clst_ass_prior = lambda x: _cluster_assignment_prior(x, clst_ass, n_mut, dirichlet_alpha)
    non_norm_log_clust_post = np.array([clst_ass_i_llh(i)+np.log(clst_ass_prior(i)) for i in range(n_clust_to_consider)])
    log_norm_val = numba_logsumexp(non_norm_log_clust_post)
    clust_post = np.exp(non_norm_log_clust_post-log_norm_val)
    new_clst_ass_i = np.random.choice(n_clust_to_consider,p=clust_post)
    # print("\tCluster assignment posterior",np.round(clust_post,decimals=4))

    if new_clst_ass_i not in clst_ass:
        #cluster has been added
        clst_ass[i] = new_clst_ass_i
        phis = np.append(phis,phi_new_cluster)
        n_clust += 1
        _update_phi(new_clst_ass_i, pairwise_occurances, n_mut, fprs, adrs, phis, clst_ass, d_rng_i, d_rng)
    clst_ass[i] = new_clst_ass_i
    if old_clst_ass_i not in clst_ass:
        #cluster has been deleted
        clst_ass[clst_ass>old_clst_ass_i] = clst_ass[clst_ass>old_clst_ass_i] - 1
        phis = np.delete(phis,old_clst_ass_i)
        n_clust -= 1
    return clst_ass, phis


def _update_phi(clst_i, pairwise_occurances, n_mut, fprs, adrs, phis, clst_ass, d_rng_i, d_rng):

    muts_in_clust_i = np.array([j for j in np.arange(n_mut) if clst_ass[j] == clst_i],dtype=np.int64)

    #Find the maximum of the function. This will be used to scale 
    # the function to allow for quadrature in non-log space.
    #In addition, we find good bounds for quadrature. Using LB=0 
    # and UB=1 would work for a normal function, but when the dataset
    # becomes large, p(phi) becomes way too peaked, and so quadrature
    # doesn't perform well. It's easy enough to find good bounds when
    # looking for the maximum value. Just say any part of the 
    # pdf < max_val-20 (or so) is essentially 0 and so set the bounds 
    # to cover the regions above max_val-20.
    to_max = lambda x: _phi_i_LLH_fraction(x, muts_in_clust_i, clst_i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng)
    llh_max, phi_max, lower_bound, upper_bound =  _find_max_and_bounds_of_logfun(to_max, 0, 1, max_threshold=-0.5, non_zero_threshold=-20, n_init_samples=20, max_iterations=100)
    
    #Perform quadrature to get the normalization factor for p(phi)
    #Discontinuity points are those where different relationships 
    # become possible or impossible. E.g., if phi_a>phi_b, then 
    # we know that b cannot be ancestral to a.
    phi_i_lh = lambda x: _phi_i_LH_fraction(x, muts_in_clust_i, clst_i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng, scale=llh_max)
    discontinuity_points = np.append(phis[(phis>lower_bound) & (phis<upper_bound)],
                                     1-phis[((1-phis)>lower_bound) & ((1-phis)<upper_bound)])
    log_norm_val = np.log(quad(phi_i_lh, lower_bound, upper_bound, limit=np.max([50,2*len(discontinuity_points)]), points=discontinuity_points)[0])+llh_max
    
    phi_post = lambda x: _phi_i_LH_fraction(x, muts_in_clust_i, clst_i, n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng, scale=log_norm_val)
    new_phi = _monte_carlo_sample_given_continuous_pdf(pdf=phi_post, max_pdf=phi_post(phi_max), lower_bound=lower_bound, upper_bound=upper_bound)
    phis[clst_i] = new_phi

    return phis


def _perform_gibbs_sampling(pairwise_occurances, n_clust, n_mut, adrs, fprs, clst_ass, phis, n_iter, burnin, dirichlet_alpha, d_rng_i, d_rng, ret_all_iters=False):

    #Aaaand I suppose start iterating through Gibb's samples
    best_llh = -np.inf
    best_phis = []
    best_pis = []
    best_nclust = n_clust
    sampled_clst_ass = []
    sampled_phis = []
    sampled_llhs = []
    sampled_nclusts = []
    for iter in range(n_iter):
        if iter % 5 == 0:
            print("Iter: {}/{}".format(iter,n_iter))
        current_llh = _p_data_given_theta_phi_and_pi(n_mut, fprs, adrs, phis, clst_ass, pairwise_occurances, d_rng_i, d_rng)
        sampled_llhs.append(current_llh)
        sampled_clst_ass.append(np.copy(clst_ass))
        sampled_phis.append(np.copy(phis))
        sampled_nclusts.append(n_clust)
        if current_llh>best_llh:
            best_llh = current_llh
            best_phis = np.copy(phis)
            best_pis = np.copy(clst_ass)
            best_nclust = n_clust
        
        # for i in range(n_mut):
        #     _update_error_rates(i, pairwise_occurances, n_mut, fprs, adrs, phis, clst_ass, d_rng_i, d_rng)
        for i in range(n_clust):
            phis = _update_phi(i, pairwise_occurances, n_mut, fprs, adrs, phis, clst_ass, d_rng_i, d_rng)
        for i in range(n_mut):
            clst_ass, phis = _update_clst_ass(i, pairwise_occurances, n_mut, n_clust, fprs, adrs, phis, clst_ass, d_rng_i, d_rng, dirichlet_alpha)

    if ret_all_iters:
        return best_phis, best_pis, best_nclust, best_llh, sampled_phis[burnin:], sampled_clst_ass[burnin:], sampled_nclusts[burnin:], sampled_llhs[burnin:]
    else:
        return best_phis, best_pis, best_nclust, best_llh


def cluster_mutations(data, fpr, adr, n_iter, burnin, dirichlet_alpha, d_rng_i, ret_all_iters=False):
    assert data.ndim == 2
    assert n_iter > burnin
    n_mut, n_cell = data.shape
    d_rng = DataRange[d_rng_i]

    pairwise_occurances, _ = determine_all_pairwise_occurance_counts(data,d_rng_i=d_rng_i)

    #initialize the parameters we're sampling
    rng = np.random.default_rng()
    if hasattr(fpr, "__len__"):
        fprs = np.copy(fpr)
    else:
        fprs = np.ones(n_mut)*fpr

    if hasattr(adr, "__len__"):
        adrs = np.copy(adr)
    else:
        adrs = np.ones(n_mut)*adr

    phis = rng.random(n_mut)

    clst_ass = np.arange(n_mut,dtype=np.int64) #For now, each mutation will start in it's own cluster
    n_clust = np.copy(n_mut)

    print("Initial values:")
    print("nClust {}".format(n_clust))
    print("ADRs {}".format(adrs))
    print("FPRs {}".format(fprs))
    print("Mut assignments {}".format(clst_ass))
    print("Cluster phis {}\n".format(phis))

    res = _perform_gibbs_sampling(pairwise_occurances, n_clust, n_mut, adrs, fprs, clst_ass, phis, n_iter, burnin, dirichlet_alpha, d_rng_i, d_rng, ret_all_iters)

    return res


def main():
    print("mutation_clusterer is not callable by itself. Import functions in run them elsewhere.")
    return

if __name__ == "__main__":
    main()