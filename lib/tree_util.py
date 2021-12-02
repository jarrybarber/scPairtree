import numpy as np
from numba import njit

@njit
def make_ancestral_from_adj(adj, check_validity=False):
    #NOTE: This code was copied from Jeff's util function.
    # --> Move to it's own util function!!

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
