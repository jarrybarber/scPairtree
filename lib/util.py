import os
import numpy as np
from numba import njit

from common import DataRange

#Should work as long as util.py is in the bin folder
LIB_DIR  = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(LIB_DIR)
DATA_DIR = os.path.join(BASE_DIR,"data")
OUT_DIR  = os.path.join(BASE_DIR,"out")
RUNS_DIR = os.path.join(BASE_DIR,"runs")

def load_data(fn, data_dir=None):
    #Loads a data file without column or row labels.
    #If there is a .mutnames, then will load that and set as mutation names
    
    if data_dir is not None:
        fn = os.path.join(data_dir,fn) 
    data = np.loadtxt(fn, dtype=np.int16)

    #Commented out. Specifying a mutids file should be explicit
    # if os.path.isfile(fn + ".mutids"):
    #     mut_ids = np.loadtxt(fn + ".mutids",dtype=str)
    # else:
    #     mut_ids = np.arange(1,data.shape[0]+1)

    return data #, mut_ids


def determine_all_cluster_pair_occurance_counts(data, clust_ass, d_rng_i):
    clusts = np.unique(clust_ass)
    n_clust = len(clusts)
    n_dtype = len(DataRange[d_rng_i])
    mut_pairwise_occurances, _ = determine_all_mutation_pair_occurance_counts(data,d_rng_i)

    clust_pairwise_occurances = np.zeros((n_dtype,n_dtype,n_clust,n_clust),dtype=int)
    for i,clst_i in enumerate(clusts):
        for j,clst_j in enumerate(clusts):
            clst_i_muts = np.flatnonzero(clust_ass==clst_i)
            clst_j_muts = np.flatnonzero(clust_ass==clst_j)
            for mut_a in clst_i_muts:
                for mut_b in clst_j_muts:
                    if mut_a==mut_b:
                        continue
                    clust_pairwise_occurances[:,:,i,j] += mut_pairwise_occurances[:,:,mut_a,mut_b]
            if clst_i == clst_j:
                clust_pairwise_occurances[range(n_dtype),range(n_dtype),i,j] = clust_pairwise_occurances[range(n_dtype),range(n_dtype),i,j] / 2
    return clust_pairwise_occurances


def determine_all_mutation_pair_occurance_counts(data, d_rng_i):
    dat_vals = DataRange[d_rng_i]
    pairwise_occurances = np.swapaxes(np.array([[determine_mutation_pair_occurance_counts(data, [i,j]) for i in dat_vals] for j in dat_vals]),0,1)

    return pairwise_occurances, dat_vals


def determine_mutation_pair_occurance_counts(data,pair_val):
    # This will take data of the form nMuts x nCells and a pair value and determine 
    # the counts of the pair value across every possible pair. E.g., how many times 
    # a [1 0] or a [0,3] occurs in the data for a mutation pair
    assert len(pair_val) == 2
    assert pair_val[0] in (0,1,2,3)
    assert pair_val[1] in (0,1,2,3)
    #First, we need separate boolean matricies for each condition
    has_1st = (data==pair_val[0]).astype(float) #Strange issue with int matricies where it takes forever to calculate. switch to float for mat mult, then switch back
    has_2nd = (data==pair_val[1]).astype(float)
    #For every mutation pair, count the number of times both occur across all cells
    count_mat = has_1st @ np.transpose(has_2nd)

    return count_mat.astype(int)



@njit(cache=True)
def seed_numba_rng(seed):
    #Numba-compiled functions use their own random number generator, which isn't seeded
    #by calling np.random.seed() from regular Python code. Calling it from within a compiled
    #function seeds numba's generator (for the current process) instead.
    np.random.seed(seed)

@njit(cache=True)
def softmax(V):
    #Note: taken from Jeff's util code
    B = np.max(V)
    log_sum = B + np.log(np.sum(np.exp(V - B)))
    log_softmax = V - log_sum
    smax = np.exp(log_softmax)
    # The vector sum will be close to 1 at this point, but not close enough to
    # make np.random.choice happy -- sometimes it will issue the error
    # "probabilities do not sum to 1" from mtrand.RandomState.choice.
    # To fix this, renormalize the vector.
    smax /= np.sum(smax)
    #assert np.isclose(np.sum(smax), 1)
    return smax

@njit(cache=True)
def isclose(a,b,atol=1e-8,rtol=1e-5):
    return np.abs(a-b)<(atol + rtol*np.abs(b))

@njit(cache=True)
def find_first(item, vec):
    """return the index of the first occurence of item in vec"""
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


@njit(cache=True)
def numba_logsumexp(V):
    #This does not support an `axis` argument. Just flattens any input vector and acts on the whole thing
    B = np.max(V)
    # Avoid NaNs when inputs are all -inf.
    # Numba doesn't support `np.isneginf`, alas.
    if np.isinf(B) and B < 0:
        return B
    summed = np.sum(np.exp(V - B))
    log_sum = B + np.log(summed)
    return log_sum