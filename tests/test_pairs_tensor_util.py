import numpy as np
import pytest

import pairs_tensor_util as ptu
from common import Models, DataRange

# Most functions here return probabilities, so for most inputs there's no
# hand-calculated "right answer" to compare against. Instead, these tests check
# properties that the maths says must always hold, e.g. that a probability
# distribution sums to 1.

DATA_RANGES = [0, 1, 2]
TREE_MODELS = [Models.A_B, Models.B_A, Models.diff_branches]


# ---- p_data_given_truth_and_errors ----

# Stacking two parametrize decorators runs every combination: 3 data ranges x
# 2 true values x 4 error rate pairs = 24 tests.
@pytest.mark.parametrize("d_rng_i", DATA_RANGES)
@pytest.mark.parametrize("t", [0, 1])
@pytest.mark.parametrize("fpr, ado", [(0.0, 0.0), (0.01, 0.2), (0.3, 0.5), (0.0, 1.0)])
def test_p_data_given_truth_sums_to_one(d_rng_i, t, fpr, ado):
    # Whatever the true genotype, the observed value must be one of the values
    # in the data range.
    total = sum(ptu.p_data_given_truth_and_errors(d, t, fpr, ado, d_rng_i)
                for d in DataRange[d_rng_i])
    assert total == pytest.approx(1)


@pytest.mark.parametrize("d_rng_i", DATA_RANGES)
@pytest.mark.parametrize("t", [0, 1])
def test_p_data_given_truth_without_errors(d_rng_i, t):
    # With no errors, the observed value is always the true value.
    for d in DataRange[d_rng_i]:
        expected = 1.0 if d == t else 0.0
        assert ptu.p_data_given_truth_and_errors(d, t, 0.0, 0.0, d_rng_i) == expected


@pytest.mark.parametrize("d_rng_i, no_reads_value", [(0, 0), (1, 3), (2, 3)])
@pytest.mark.parametrize("t", [0, 1])
def test_p_data_given_truth_with_total_dropout(d_rng_i, no_reads_value, t):
    # If both alleles always drop out, no reads are ever seen. Data range 0 has
    # no "no reads" value, so this shows up as "no variant detected" (0).
    p = ptu.p_data_given_truth_and_errors(no_reads_value, t, 0.01, 1.0, d_rng_i)
    assert p == pytest.approx(1)


# ---- p_trueDat_given_model_and_phis ----

# (model, phi1, phi2), with phis chosen to be consistent with each model.
CONSISTENT_PHIS = [
    pytest.param(Models.A_B, 0.7, 0.3, id="A_B"),
    pytest.param(Models.A_B, 1.0, 0.0, id="A_B-extreme"),
    pytest.param(Models.B_A, 0.3, 0.7, id="B_A"),
    pytest.param(Models.diff_branches, 0.2, 0.5, id="branched"),
    pytest.param(Models.diff_branches, 0.5, 0.5, id="branched-sum-is-1"),
    pytest.param(Models.cocluster, 0.4, 0.4, id="cocluster"),
    pytest.param(Models.garbage, 0.4, 0.9, id="garbage"),
]


@pytest.mark.parametrize("model, phi1, phi2", CONSISTENT_PHIS)
def test_p_trueDat_is_a_distribution(model, phi1, phi2):
    # P[t1, t2] = probability that a cell has true genotypes t1 and t2.
    P = np.array([[ptu.p_trueDat_given_model_and_phis(t1, t2, model, phi1, phi2)
                   for t2 in (0, 1)] for t1 in (0, 1)])
    assert np.all(P >= 0)
    assert P.sum() == pytest.approx(1)


@pytest.mark.parametrize("model, phi1, phi2", CONSISTENT_PHIS)
def test_p_trueDat_marginals_equal_phis(model, phi1, phi2):
    # phi is the fraction of cells carrying a mutation, so whatever the model,
    # P(t1 = 1) must equal phi1 and P(t2 = 1) must equal phi2.
    P = np.array([[ptu.p_trueDat_given_model_and_phis(t1, t2, model, phi1, phi2)
                   for t2 in (0, 1)] for t1 in (0, 1)])
    assert P[1, :].sum() == pytest.approx(phi1)
    assert P[:, 1].sum() == pytest.approx(phi2)


# ---- p_phi_given_model ----

@pytest.mark.parametrize("model", TREE_MODELS + [Models.garbage])
def test_p_phi_given_model_integrates_to_one(model):
    # p_phi_given_model is a probability density over the unit square, so it
    # must integrate to 1. Approximate the integral by averaging over a grid of
    # points (each grid square has area 1/n^2, and there are n^2 of them).
    # Cocluster is skipped because its density lies on a line, which a grid
    # can't sample.
    n = 400
    grid = (np.arange(n) + 0.5) / n
    integral = np.mean([ptu.p_phi_given_model(model, a, b) for a in grid for b in grid])
    assert integral == pytest.approx(1, abs=0.01)


@pytest.mark.parametrize("model", TREE_MODELS + [Models.garbage, Models.cocluster])
@pytest.mark.parametrize("phi_a, phi_b", [(-0.1, 0.5), (0.5, -0.1), (1.1, 0.5), (0.5, 1.1)])
def test_p_phi_given_model_is_zero_outside_unit_square(model, phi_a, phi_b):
    assert ptu.p_phi_given_model(model, phi_a, phi_b) == 0


@pytest.mark.parametrize("model, phi_a, phi_b", [
    pytest.param(Models.A_B, 0.3, 0.7, id="A_B-child-has-more-cells"),
    pytest.param(Models.B_A, 0.7, 0.3, id="B_A-child-has-more-cells"),
    pytest.param(Models.diff_branches, 0.6, 0.6, id="branched-sum-over-1"),
    pytest.param(Models.cocluster, 0.3, 0.7, id="cocluster-different-phis"),
])
def test_p_phi_given_model_is_zero_for_impossible_phis(model, phi_a, phi_b):
    # e.g. if A is an ancestor of B, every cell with B also has A, so A must be
    # in at least as many cells as B.
    assert ptu.p_phi_given_model(model, phi_a, phi_b) == 0


# ---- p_model_given_phi ----

@pytest.mark.parametrize("phi_a, phi_b", [(0.7, 0.3), (0.3, 0.7), (0.2, 0.5), (0.9, 0.6), (1.0, 0.0)])
def test_p_model_given_phi_sums_to_one_over_tree_models(phi_a, phi_b):
    # Off the diagonal (phi_a != phi_b), exactly one of ancestral, descendant
    # or branched applies, or two apply with equal probability.
    total = sum(ptu.p_model_given_phi(m, phi_a, phi_b) for m in TREE_MODELS)
    assert total == pytest.approx(1)


# ---- log_p_data_given_model_phis_and_errors ----

@pytest.mark.parametrize("model, phi_a, phi_b", [
    pytest.param(Models.A_B, 0.7, 0.3, id="A_B"),
    pytest.param(Models.diff_branches, 0.2, 0.5, id="branched"),
    pytest.param(Models.cocluster, 0.4, 0.4, id="cocluster"),
    pytest.param(Models.garbage, 0.4, 0.9, id="garbage"),
])
@pytest.mark.parametrize("d_rng_i", DATA_RANGES)
def test_single_cell_data_probabilities_sum_to_one(model, phi_a, phi_b, d_rng_i):
    # With a dataset of one cell, the pair counts are all 0 except a single 1
    # for the observed pair of values. Summing the probability of every
    # possible one-cell dataset must give 1. The function includes the phi
    # prior, so that's divided back out.
    n = len(DataRange[d_rng_i])
    log_prior = np.log(ptu.p_phi_given_model(model, phi_a, phi_b))
    total = 0
    for i in range(n):
        for j in range(n):
            counts = np.zeros((n, n))
            counts[i, j] = 1
            log_p = ptu.log_p_data_given_model_phis_and_errors(
                model, counts, 0.03, 0.05, 0.2, 0.1, phi_a, phi_b, d_rng_i)
            total += np.exp(log_p - log_prior)
    assert total == pytest.approx(1)


def test_log_p_data_with_no_data_is_log_prior():
    counts = np.zeros((3, 3))
    log_p = ptu.log_p_data_given_model_phis_and_errors(
        Models.A_B, counts, 0.03, 0.05, 0.2, 0.1, 0.7, 0.3, 1)
    assert log_p == pytest.approx(np.log(ptu.p_phi_given_model(Models.A_B, 0.7, 0.3)))


def test_log_p_data_is_neginf_for_impossible_phis():
    counts = np.ones((3, 3))
    log_p = ptu.log_p_data_given_model_phis_and_errors(
        Models.A_B, counts, 0.03, 0.05, 0.2, 0.1, 0.3, 0.7, 1)
    assert log_p == -np.inf


def test_log_p_data_with_zero_dropout_is_finite():
    # With ado=0, the 'no reads' value has probability 0. This used to give
    # count * log(0) = 0 * -inf = nan even where the count was 0.
    counts = np.array([[5, 0, 0],
                       [0, 5, 0],
                       [0, 0, 0]])
    log_p = ptu.log_p_data_given_model_phis_and_errors(
        Models.A_B, counts, 0.01, 0.01, 0.0, 0.0, 0.7, 0.3, 1)
    assert np.isfinite(log_p)
