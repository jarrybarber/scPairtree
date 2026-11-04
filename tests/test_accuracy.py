import numpy as np
import pytest

from scpairtree.cli import run
from scpairtree.result_serializer import Results
from scpairtree.data_simulator import generate_simulated_data
from scpairtree import tree_util

# End-to-end accuracy tests: simulate data from a known tree, run scPairtree,
# and check that it recovers something close to the truth. Unlike the other
# tests, these can't demand an exact answer, since noisy data doesn't always
# pin down the true tree. Instead they check that accuracy stays above a
# threshold set well below what scPairtree normally achieves, so they fail if
# a change makes scPairtree meaningfully worse, but not from ordinary noise.

N_MUT = 30
N_CELL = 300
TRUE_FPR = 0.001
TRUE_ADR = 0.2
D_RNG_I = 1


# params makes the fixture itself parametrized: every test that uses it runs
# once per seed. Combined with scope="module", scPairtree runs once per seed
# and all tests in this file share each run, rather than each test running
# the (slow) pipeline itself.
@pytest.fixture(scope="module", params=[0, 1, 2])
def simulated_run(request, tmp_path_factory):
    seed = request.param
    np.random.seed(seed)
    # n_clust = n_mut gives each tree node exactly one mutation.
    data, (_, true_adj, _, mut_assignments) = generate_simulated_data(
        n_clust=N_MUT, n_cell=N_CELL, n_mut=N_MUT, FPR=TRUE_FPR, ADR=TRUE_ADR,
        cell_alpha=1, mut_alpha=1, drange=D_RNG_I)

    run_dir = tmp_path_factory.mktemp(f"seed{seed}")
    data_fn = str(run_dir / "genotypes.txt")
    results_fn = str(run_dir / "results")
    np.savetxt(data_fn, data, "%d")

    # tree_chains and parallel are set explicitly because their defaults depend
    # on the number of CPU cores, which would make results differ by machine.
    run(data_fn, D_RNG_I, results_fn, seed=seed, tree_chains=4, parallel=2,
        trees_per_chain=3000)

    # Express the true tree with one node per mutation, as scPairtree does.
    true_mut_adj = tree_util.convert_clust_mat_to_mut_mat(true_adj, mut_assignments)
    return Results(results_fn), true_mut_adj


def relationship_accuracy(adj_a, adj_b):
    # Fraction of node pairs that have the same relationship (ancestor,
    # descendant or different branches) in both trees.
    rels_a = tree_util.compute_node_relations(adj_a)
    rels_b = tree_util.compute_node_relations(adj_b)
    pairs = np.triu_indices(len(adj_a), 1)
    return np.mean(rels_a[pairs] == rels_b[pairs])


# @pytest.mark.slow labels these tests (see pytest.ini) so they can be skipped
# with `pytest -m "not slow"`.
@pytest.mark.slow
def test_best_tree_is_close_to_true_tree(simulated_run):
    results, true_adj = simulated_run
    # scPairtree typically gets over 98% of relationships right on this data.
    assert relationship_accuracy(results.get("best_tree_adj"), true_adj) >= 0.95


@pytest.mark.slow
def test_estimated_error_rates_are_close_to_true_rates(simulated_run):
    results, _ = simulated_run
    assert np.median(results.get("est_ADOs")) == pytest.approx(TRUE_ADR, abs=0.05)
    assert np.median(results.get("est_FPRs")) < 0.01
