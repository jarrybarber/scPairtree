import os
import numpy as np
from numba import njit
from scipy.special import loggamma, logsumexp

#Should work as long as util.py is in the bin folder
LIB_DIR  = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(LIB_DIR)
DATA_DIR = os.path.join(BASE_DIR,"data")
OUT_DIR  = os.path.join(BASE_DIR,"out")
RUNS_DIR = os.path.join(BASE_DIR,"runs")


def load_sim_data(fn):
    #convert output to pandas?
    data = []
    ssm_ids = []
    count = 0
    to_open = os.path.join(DATA_DIR,"simulated",fn)
    with open(to_open,'r') as f:
        for row in f.readlines():
            entries = row.replace("\n","").split("\t")
            data.append([int(i) for i in entries])
            ssm_ids.append('s' + str(count))
            count += 1
    return np.array(data), np.array(ssm_ids)


def determine_pairwise_occurance_counts(data):
    # This will take data of the form nSSMs x nCells and determine the counts of
    # each pairwise occurance. I.e., how many times a [1 1], [1 0], [0 1], [0 0]
    # occurs in the data for each SSM pair. Values other than 0 or 1 (eg, 3= no 
    # call) will be omitted.

    #First, need separate boolean matricies for each condition
    has_ref = (data==0).astype(int) + 0.0 #Strange issue when matricies get large... takes forever when integers as well... switch to float for mat mult, then switch back
    has_alt = (data==1).astype(int) + 0.0
    #For every mutation pair, count the number of times both occur across all cells
    n11 = has_alt @ np.transpose(has_alt)
    #For every mutation pair, count the number of times neither occur across all cells
    n00 = has_ref @ np.transpose(has_ref)
    #For every mutation pair, count the number of times only first occurs across all cells
    n10 = has_alt @ np.transpose(has_ref)
    #For every mutation pair, count the number of times only second occurs across all cells
    n01 = has_ref @ np.transpose(has_alt)

    return n11.astype(int),n10.astype(int),n01.astype(int),n00.astype(int)


def calc_tensor_prob(tensor):
    #Should be 3 axis: 1st for models, 2nd and 3rd for SNV comps
    #I assume all values are logged.
    assert not np.any(tensor==0)
    return np.sum(logsumexp(tensor,axis=0))



##OLD STUFF FROM BEFORE QUADRATURE###
def log_bec(n,i,j):
    #binomial expansion coefficient - logged form:
    return loggamma(n+1) - loggamma(i+1) - loggamma(j+1)


def log_tec(n,i,j,k):
    return loggamma(n+1) - loggamma(i+1) - loggamma(j+1) - loggamma(k+1)


def log_qec(n,i,j,k,m):
    return loggamma(n+1) - loggamma(i+1) - loggamma(j+1) - loggamma(k+1) - loggamma(m+1)


# From Jeff <3 (Not used at the moment. Still using scipy for logsumexp and loggamma)
# @njit
# def logsumexp(V, axis=None):
#     B = np.max(V)
#     # Explicitly checking `axis` is necessary for Numba, which doesn't support
#     # `axis=None` in calling `np.sum()`.
#     if axis is None:
#       # Avoid NaNs when inputs are all -inf.
#       # Numba doesn't support `np.isneginf`, alas.
#       if np.isinf(B) and B < 0:
#         return B
#       summed = np.sum(np.exp(V - B))
#     else:
#       # NB: this is suboptimal, since we should call `np.max(V, axis)`, but Numba
#       # doesn't yet support the axis argument. So, we end up using the scalar
#       # maximum across the entire array, not the vector maximum across the axis.
#       #
#       # NB part deux: if all the elements across one axis of the array are -inf,
#       # this will break and return NaN, when it should instead return -inf. So
#       # long as `np.max(..., axis)` isn't supported in Numba, this is non-trivial
#       # to fix.
#       summed = np.sum(np.exp(V - B), axis)
#     log_sum = B + np.log(summed)
#     return log_sum
