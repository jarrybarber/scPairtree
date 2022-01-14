import os
import numpy as np
from numba import njit
from scipy.special import loggamma, logsumexp
from collections import namedtuple

from common import Models

#Note, this may work better as a pandas object so that any sorting of 
#snvs and cells automatically works when sorting data matrix.
_Data = namedtuple('_Data', (
  'data',
  'n_snvs',
  'n_cells',
  'snv_ids',
  #'cell_ids' #Note implemented yet
))

#Should work as long as util.py is in the bin folder
LIB_DIR  = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(LIB_DIR)
DATA_DIR = os.path.join(BASE_DIR,"data")
OUT_DIR  = os.path.join(BASE_DIR,"out")
RUNS_DIR = os.path.join(BASE_DIR,"runs")


def load_sim_data(fn):
    data = []
    snv_ids = []
    count = 0
    to_open = os.path.join(DATA_DIR,"simulated",fn)
    with open(to_open,'r') as f:
        for row in f.readlines():
            entries = row.replace("\n","").split("\t")
            data.append([int(i) for i in entries])
            snv_ids.append('s' + str(count))
            count += 1
    return _Data(data=np.array(data),n_snvs=len(data),n_cells=len(data[0]),snv_ids=snv_ids)


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

@njit
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

@njit
def make_ancestral_from_adj(adj, check_validity=False):
    #Note: taken from Jeff's util code.
    K = len(adj)
    root = 0

    if check_validity:
        # By default, disable checks to improve performance.
        assert np.all(1 == np.diag(adj))
        expected_sum = 2 * np.ones(K)
        expected_sum[root] = 1
        assert np.array_equal(expected_sum, np.sum(adj, axis=0))

    Z = np.copy(adj)
    np.fill_diagonal(Z, 0)
    stack = [root]
    while len(stack) > 0:
        P = stack.pop()
        C = np.flatnonzero(Z[P])
        if len(C) == 0:
            continue
        # Set ancestors of `C` to those of their parent `P`.
        C_anc = np.copy(Z[:,P])
        C_anc[P] = 1
        # Turn `C_anc` into column vector.
        Z[:,C] = np.expand_dims(C_anc, 1)
        stack += list(C)
    np.fill_diagonal(Z, 1)

    if check_validity:
        assert np.array_equal(Z[root], np.ones(K))
    return Z

@njit
def compute_node_relations(adj, check_validity=False):
    #Note: taken from Jeff's util code.
    #May make sense to move somewhere else... Perhaps some tree or pairs tensor util.
    K = len(adj)
    anc = make_ancestral_from_adj(adj, check_validity)
    np.fill_diagonal(anc, 0)

    R = np.full((K, K), Models.diff_branches, dtype=np.int8)
    for idx in range(K):
        R[idx][anc[idx]   == 1] = Models.A_B
        R[idx][anc[:,idx] == 1] = Models.B_A
    np.fill_diagonal(R, Models.cocluster)

    if check_validity:
        assert np.all(R[0]   == Models.A_B)
        assert np.all(R[:,0] == Models.B_A)
    return R

def remove_rowcol(arr, indices):
    #NOTE: taken from Jeff's util code
    '''Remove rows and columns at `indices`.'''
    # Calling `np.delete` with `indices` as an empty array produces an exception
    # in Python 3.8 but not 3.7.
    if len(indices) == 0:
        return np.copy(arr)
    # Using a mask requires only creating a new array once (and then presumably
    # another array to apply the mask). Calling np.delete() multiple times would
    # create separate arrays for every element in `indices`.
    shape = list(arr.shape)
    # Must convert to list *before* array, as indices is often a set, and
    # converting a set directly to an array yields a dtype of `object` that
    # contains the set. It's really weird.
    indices = np.array(list(indices))
    # Only check the first two dimensions, as we ignore all others.
    assert np.all(0 <= indices) and np.all(indices < min(shape[:2]))

    for axis in (0, 1):
        arr = np.delete(arr, indices, axis=axis)
        shape[axis] -= len(indices)

    assert np.array_equal(arr.shape, shape)
    return arr

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
