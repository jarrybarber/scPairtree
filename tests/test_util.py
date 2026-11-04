import numpy as np
import pytest
from numba import njit
from scipy.special import logsumexp

import util


# ---- find_first ----

def test_find_first_returns_index_of_item():
    assert util.find_first(7, np.array([3, 7, 5])) == 1


def test_find_first_returns_first_of_duplicates():
    assert util.find_first(7, np.array([7, 3, 7])) == 0


def test_find_first_returns_minus_one_when_missing():
    assert util.find_first(9, np.array([3, 7, 5])) == -1


# ---- softmax ----

def test_softmax_matches_hand_calculation():
    # exp(0) = 1 and exp(log 2) = 2, so the result should be [1/3, 2/3].
    smax = util.softmax(np.array([0.0, np.log(2)]))
    assert smax == pytest.approx([1/3, 2/3])


def test_softmax_sums_to_one():
    rng = np.random.default_rng(0)
    smax = util.softmax(rng.normal(size=20))
    assert np.sum(smax) == pytest.approx(1)


def test_softmax_handles_large_inputs():
    # A naive exp(1000) overflows to inf. softmax subtracts the max first to
    # avoid this, so shifting every input by the same amount shouldn't matter.
    smax = util.softmax(np.array([1000.0, 1000.0 + np.log(2)]))
    assert smax == pytest.approx([1/3, 2/3])


# ---- numba_logsumexp ----

def test_numba_logsumexp_matches_scipy():
    rng = np.random.default_rng(0)
    V = rng.normal(size=20)
    assert util.numba_logsumexp(V) == pytest.approx(logsumexp(V))


def test_numba_logsumexp_handles_large_inputs():
    V = np.array([1000.0, 1000.0])
    assert util.numba_logsumexp(V) == pytest.approx(1000 + np.log(2))


def test_numba_logsumexp_all_neginf():
    # log(0 + 0) = -inf. Without the special case this would return nan.
    V = np.array([-np.inf, -np.inf])
    assert util.numba_logsumexp(V) == -np.inf


# ---- determine_mutation_pair_occurance_counts ----

# 3 mutations (rows) x 4 cells (columns).
PAIR_DATA = np.array([
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 0, 1],
])


def test_pair_counts_for_1_1():
    # counts[a,b] = number of cells where mutation a is 1 AND mutation b is 1.
    # e.g. mutations 0 and 1 are both 1 only in cell 0, so counts[0,1] = 1.
    counts = util.determine_mutation_pair_occurance_counts(PAIR_DATA, [1, 1])
    expected = np.array([
        [2, 1, 0],
        [1, 2, 0],
        [0, 0, 1],
    ])
    assert np.array_equal(counts, expected)


def test_pair_counts_for_1_0():
    # counts[a,b] = number of cells where mutation a is 1 AND mutation b is 0.
    # Unlike [1,1], this isn't symmetric: counts[0,2] = 2 but counts[2,0] = 1.
    counts = util.determine_mutation_pair_occurance_counts(PAIR_DATA, [1, 0])
    expected = np.array([
        [0, 1, 2],
        [1, 0, 2],
        [1, 1, 0],
    ])
    assert np.array_equal(counts, expected)


# ---- load_data ----

def test_load_data(tmp_path):
    # tmp_path is a fresh, empty folder that pytest creates for this test only.
    fn = tmp_path / "genotypes.txt"
    fn.write_text("0 1 3\n1\t0 1\n")  # mixed spaces and tabs are allowed

    data = util.load_data(str(fn))
    assert np.array_equal(data, [[0, 1, 3], [1, 0, 1]])


def test_load_data_with_data_dir(tmp_path):
    (tmp_path / "genotypes.txt").write_text("0 1\n1 0\n")

    data = util.load_data("genotypes.txt", data_dir=str(tmp_path))
    assert np.array_equal(data, [[0, 1], [1, 0]])


# ---- seed_numba_rng ----

@njit
def _numba_random():
    return np.random.random()


def test_seed_numba_rng_makes_numba_draws_repeatable():
    util.seed_numba_rng(1)
    first = _numba_random()
    util.seed_numba_rng(1)
    second = _numba_random()
    assert first == second


def test_np_random_seed_does_not_seed_numba():
    # The reason seed_numba_rng exists: numba-compiled code has its own random
    # number generator, which np.random.seed() called from Python doesn't affect.
    np.random.seed(1)
    first = _numba_random()
    np.random.seed(1)
    second = _numba_random()
    assert first != second
