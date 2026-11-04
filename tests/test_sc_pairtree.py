import numpy as np
import pytest

from sc_pairtree import run
from result_serializer import Results
from data_simulator import generate_simulated_data

# These tests call run(), the same function the command line uses, so they
# exercise the whole pipeline on a tiny simulated dataset.

D_RNG_I = 1

# Results entries that should be identical between runs with the same seed.
# Runtimes are left out since they change from run to run.
SEEDED_RESULTS = [
    "est_FPRs",
    "est_ADOs",
    "pairs_tensor",
    "adj_mats",
    "tree_llhs",
    "dfpt_samples",
    "dfpt_sample_log_probs",
]


# A fixture is a function that prepares something tests need. A test requests
# it by naming it as a parameter, and pytest calls the fixture and passes in
# the return value. scope="module" means it runs once and every test in this
# file shares the result, rather than re-simulating data for each test.
@pytest.fixture(scope="module")
def data_fn(tmp_path_factory):
    np.random.seed(0)
    data, _ = generate_simulated_data(n_clust=6, n_cell=60, n_mut=6, FPR=0.001, ADR=0.2,
                                      cell_alpha=1, mut_alpha=1, drange=D_RNG_I)
    fn = tmp_path_factory.mktemp("data") / "genotypes.txt"
    np.savetxt(fn, data, "%d")
    return str(fn)


def run_and_load(data_fn, results_fn, **kwargs):
    run(data_fn, D_RNG_I, str(results_fn), tree_chains=2, trees_per_chain=200,
        perform_dfpt_sampling=True, dfpt_nsamples=200, **kwargs)
    results = Results(str(results_fn))
    return {name: results.get(name) for name in SEEDED_RESULTS}


# ---- Reproducibility ----

# parallel=0 runs the MCMC chains one after another in this process, while
# parallel=2 runs them in separate worker processes. Each worker has its own
# random number generators, so seeding needs to work in both cases.
@pytest.mark.parametrize("parallel", [0, 2])
def test_same_seed_gives_same_results(data_fn, tmp_path, parallel):
    first = run_and_load(data_fn, tmp_path / "first", seed=5, parallel=parallel)
    second = run_and_load(data_fn, tmp_path / "second", seed=5, parallel=parallel)
    for name in SEEDED_RESULTS:
        assert np.array_equal(first[name], second[name]), f"{name} differs between runs"


def test_different_seeds_give_different_results(data_fn, tmp_path):
    # Without this test, the one above would also pass if the pipeline used no
    # randomness at all, or if the seed were ignored and replaced by a constant.
    first = run_and_load(data_fn, tmp_path / "first", seed=5, parallel=0)
    second = run_and_load(data_fn, tmp_path / "second", seed=6, parallel=0)
    assert not np.array_equal(first["adj_mats"], second["adj_mats"])


# ---- Argument checking ----

@pytest.mark.parametrize("fpr, adr", [
    (-0.1, 0.2),
    (1.5, 0.2),
    (0.0, 0.2),
    (0.01, -0.1),
    (0.01, 1.5),
    (0.01, 1.0),
])
def test_error_rates_outside_0_1_are_rejected(data_fn, tmp_path, fpr, adr):
    # pytest.raises passes only if the code inside the `with` block raises a
    # ValueError, and `match` checks the error message.
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        run(data_fn, D_RNG_I, str(tmp_path / "results"), fpr=fpr, adr=adr)


def test_only_one_error_rate_is_rejected(data_fn, tmp_path):
    with pytest.raises(ValueError, match="both error rates"):
        run(data_fn, D_RNG_I, str(tmp_path / "results"), fpr=0.01)


def test_unknown_argument_is_rejected(data_fn, tmp_path):
    with pytest.raises(NameError, match="Did not recognize"):
        run(data_fn, D_RNG_I, str(tmp_path / "results"), not_an_argument=1)
