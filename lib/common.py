#This will house various constants and structures used by the program
import numpy as np
from numba import njit
from collections import namedtuple

_LOGEPSILON = -30
_EPSILON    = np.exp(_LOGEPSILON)

#Using the same indices as Jeff to help keep things clean.
_ModelChoice = namedtuple('_ModelChoice', (
  'garbage',
  'cocluster',
  'A_B',
  'B_A',
  'diff_branches',
))
Models = _ModelChoice(garbage=0, cocluster=1, A_B=2, B_A=3, diff_branches=4)
NUM_MODELS = len(Models)
ALL_MODELS = _ModelChoice._fields

_DataType = namedtuple('_Datatype', (
  'var_notvar',
  'ref_var_nodata',
  'ref_hetvar_homvar_nodata'
))
DataRangeIdx = _DataType(var_notvar=0, ref_var_nodata=1, ref_hetvar_homvar_nodata=2)
DataRange = ((0,1),(0,1,3),(0,1,2,3))

@njit(cache=True)
def get_d_range(d_rng_i):
  #Numba is really really annoying when it comes to any type of list embedded in lists.
	#So, I decided to just make this function to allow me to get the d_range from d_range_index
	#Otherwise I have to pass both the list and the index all over the place.
    if d_rng_i == 0:
        d_range = [0,1]
    elif d_rng_i == 1:
        d_range = [0,1,3]
    elif d_rng_i == 2:
        d_range = [0,1,2,3]
    return d_range

def debug(*args, **kwargs):
    if hasattr(debug, 'DEBUG') and debug.DEBUG:
        print(*args, **kwargs)


def ensure_valid_tree(adj):
  #NOTE: Taken from Jeff's common.py
  # I had several issues with subtle bugs in my tree initialization algorithm
  # creating invalid trees. This function is useful to ensure that `adj`
  # corresponds to a valid tree.
  adj = np.copy(adj)
  K = len(adj)
  assert np.all(np.diag(adj) == 1)
  np.fill_diagonal(adj, 0)
  visited = set()

  stack = [0]
  while len(stack) > 0:
    P = stack.pop()
    assert P not in visited
    visited.add(P)
    C = list(np.flatnonzero(adj[P]))
    stack += C
  assert visited == set(range(K))