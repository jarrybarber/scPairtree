import os
import numpy as np
from numba import njit
from scipy.special import loggamma, logsumexp
from collections import namedtuple

from common import Models, DataRangeIdx, DataRange, _EPSILON

#Note, this may work better as a pandas object so that any sorting of 
#snvs and cells automatically works when sorting data matrix.
_Data = namedtuple('_Data', (
  'data',
  'n_snvs',
  'n_cells',
  'snv_ids',
  #'cell_ids' #Not implemented yet
))

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

    if os.path.isfile(fn + ".mutnames"):
        mut_names = np.loadtxt(fn + ".mutnames",dtype=str)
    else:
        mut_names = np.arange(1,data.shape[0]+1)

    return data, mut_names


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

def determine_all_pairwise_occurance_counts(data, d_rng_i):
    dat_vals = DataRange[d_rng_i]
    pairwise_occurances = np.swapaxes(np.array([[determine_pairwise_occurance_counts(data, [i,j]) for i in dat_vals] for j in dat_vals]),0,1)

    return pairwise_occurances, dat_vals


def determine_pairwise_occurance_counts(data,pair_val):
    # This will take data of the form nMuts x nCells and a pair value and determine 
    # the counts of the pair value across every possible pair. E.g., how many times 
    # a [1 0] or a [0,3] occurs in the data for a mutation pair
    assert len(pair_val)==2
    assert pair_val[0] in (0,1,2,3)
    assert pair_val[1] in (0,1,2,3)
    #First, we need separate boolean matricies for each condition
    has_1st = (data==pair_val[0]).astype(float) #Strange issue with int matricies where it takes forever to calculate. switch to float for mat mult, then switch back
    has_2nd = (data==pair_val[1]).astype(float)
    #For every mutation pair, count the number of times both occur across all cells
    count_mat = has_1st @ np.transpose(has_2nd)

    return count_mat.astype(int)

@njit(cache=True)
def convert_adjmatrix_to_ancmatrix(adj, check_validity=False):
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
    # np.fill_diagonal(Z, 0)
    for i in range(Z.shape[0]):
        Z[i,i] = 0
        
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
    
    # np.fill_diagonal(Z, 1)
    for i in range(Z.shape[0]):
        Z[i,i] = 1
        
    if check_validity:
        assert np.array_equal(Z[root], np.ones(K))
    return Z


@njit(cache=True)
def convert_ancmatrix_to_adjmatrix(anc):
    this_anc = np.copy(anc)
    for i in range(this_anc.shape[0]):
        this_anc[i,i] = 0
    
    adj = np.zeros(this_anc.shape, dtype=np.int8)
    for i in range(adj.shape[0]):
        adj[i,i] = 1
    
    n_clust = len(this_anc)
    for child in range(1,n_clust):
        is_anc_to_child = np.argwhere(this_anc[:,child]).flatten()
        par = 0
        for j in is_anc_to_child:
            is_dec_cur_par = this_anc[par,j]
            if is_dec_cur_par:
                par = j
        adj[par,child] = 1
    return adj


@njit(cache=True)
def convert_ancmatrix_to_parents(anc):
    n_clust = len(anc)
    this_anc = np.copy(anc)
    for i in range(n_clust):
        this_anc[i,i] = 0
    
    parents = np.zeros(n_clust-1, dtype=np.int32)
    for child in range(1,n_clust):
        is_anc_to_child = np.flatnonzero(this_anc[:,child])# np.argwhere(this_anc[:,child]).flatten()
        parent = 0
        for j in is_anc_to_child:
            is_dec_cur_par = this_anc[parent,j]
            if is_dec_cur_par:
                parent = j
        parents[child-1] = parent
    return parents


@njit(cache=True)
def convert_parents_to_ancmatrix(parents):
    adj = convert_parents_to_adjmatrix(parents)
    anc = convert_adjmatrix_to_ancmatrix(adj)
    return anc


#Taken from Jeff's Pairtree
@njit(cache=True)
def convert_parents_to_adjmatrix(parents):
    K = len(parents) + 1
    adjm = np.eye(K, dtype=np.int8)
    for i,parent in enumerate(parents):
        adjm[parent,i+1] = 1
    return adjm


#Taken from Jeff's Pairtree
@njit(cache=True)
def convert_adjmatrix_to_parents(adj):
    adj = np.copy(adj)
    np.fill_diagonal(adj, 0)
    parents = np.zeros(adj.shape[1]-1, dtype=np.int32)
    for i in range(1,adj.shape[1]):
        parents[i-1] = find_first(1,adj[:,i])
    return parents#np.argmax(adj[:,1:], axis=0)


def convert_nodeadj_to_mutadj(node_adj, mut_assignments):
    assert len(mut_assignments) == node_adj.shape[0]-1
    mutadj = np.zeros(node_adj.shape,dtype=int)
    mut_assignments = np.append(0,mut_assignments)
    node_assignments = np.zeros(mut_assignments.shape,dtype=int)
    for i,a in enumerate(mut_assignments):
        node_assignments[a] = i
    for par, chld in np.argwhere(node_adj):
        mutadj[node_assignments[par], node_assignments[chld]] = 1
    return mutadj


def convert_mutadj_to_nodeadj(mut_adj, mut_assignments):
    assert len(mut_assignments) == mut_adj.shape[0]-1
    node_adj = np.zeros(mut_adj.shape,dtype=int)
    mut_assignments = np.append(0,mut_assignments)
    for par, chld in np.argwhere(mut_adj):
        node_adj[mut_assignments[par], mut_assignments[chld]] = 1
    return node_adj


@njit(cache=True)
def compute_node_relations(adj, check_validity=False):
    #Note: taken from Jeff's util code.
    #May make sense to move somewhere else... Perhaps some tree or pairs tensor util.
    K = len(adj)
    anc = convert_adjmatrix_to_ancmatrix(adj, check_validity)
    # np.fill_diagonal(anc, 0)
    for i in range(anc.shape[0]): #better for numba
        anc[i,i] = 0

    R = np.full((K, K), Models.diff_branches, dtype=np.int8)
    for i in range(K):
        for j in range(K):
            if anc[i,j] == 1:
                R[i,j] = Models.A_B
            elif anc[j,i] == 1:
                R[i,j] = Models.B_A
                
    for i in range(K):
        R[i,i] = Models.cocluster

    if check_validity:
        assert np.all(R[0]   == Models.A_B)
        assert np.all(R[:,0] == Models.B_A)
    return R

def calc_tensor_prob(tensor):
    #Should be 3 axis: 1st for models, 2nd and 3rd for SNV comps
    #I assume all values are logged.
    assert not np.any(tensor==0)
    return np.sum(logsumexp(tensor,axis=0))

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
@njit(cache=True)
def logsumexp(V, axis=None):
    B = np.max(V)
    # Explicitly checking `axis` is necessary for Numba, which doesn't support
    # `axis=None` in calling `np.sum()`.
    if axis is None:
      # Avoid NaNs when inputs are all -inf.
      # Numba doesn't support `np.isneginf`, alas.
      if np.isinf(B) and B < 0:
        return B
      summed = np.sum(np.exp(V - B))
    else:
      # NB: this is suboptimal, since we should call `np.max(V, axis)`, but Numba
      # doesn't yet support the axis argument. So, we end up using the scalar
      # maximum across the entire array, not the vector maximum across the axis.
      #
      # NB part deux: if all the elements across one axis of the array are -inf,
      # this will break and return NaN, when it should instead return -inf. So
      # long as `np.max(..., axis)` isn't supported in Numba, this is non-trivial
      # to fix.
      summed = np.sum(np.exp(V - B), axis)
      B = np.ones(summed.shape)
    log_sum = B + np.log(summed)
    return log_sum
