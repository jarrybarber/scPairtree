import numpy as np
import pytest

from scpairtree import tree_util
from scpairtree.common import Models

# A small hand-built tree used by every test in this file. Node 0 is the root.
#
#        0
#       / \
#      1   4
#     / \
#    2   3
#
# The tree is written out in each of the three formats scPairtree uses, so the
# tests can check that converting between formats gives the expected answer.

# parents[i] is the parent of node i+1 (the root has no parent).
PARENTS = np.array([0, 1, 1, 0])

# ADJ[i,j] = 1 if i is the parent of j. The diagonal is 1 by convention.
ADJ = np.array([
    [1, 1, 0, 0, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
])

# ANC[i,j] = 1 if i is an ancestor of j. The diagonal is 1 by convention.
ANC = np.array([
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
])


def test_parents_to_adjmatrix():
    adj = tree_util.convert_parents_to_adjmatrix(PARENTS)
    assert np.array_equal(adj, ADJ)


def test_adjmatrix_to_parents():
    parents = tree_util.convert_adjmatrix_to_parents(ADJ)
    assert np.array_equal(parents, PARENTS)


def test_adjmatrix_to_ancmatrix():
    anc = tree_util.convert_adjmatrix_to_ancmatrix(ADJ)
    assert np.array_equal(anc, ANC)


def test_ancmatrix_to_adjmatrix():
    adj = tree_util.convert_ancmatrix_to_adjmatrix(ANC)
    assert np.array_equal(adj, ADJ)


def test_ancmatrix_to_parents():
    parents = tree_util.convert_ancmatrix_to_parents(ANC)
    assert np.array_equal(parents, PARENTS)


def test_parents_to_ancmatrix():
    anc = tree_util.convert_parents_to_ancmatrix(PARENTS)
    assert np.array_equal(anc, ANC)


def test_compute_node_relations():
    rels = tree_util.compute_node_relations(ADJ)
    assert rels[1, 2] == Models.A_B            # 1 is an ancestor of 2
    assert rels[2, 1] == Models.B_A            # 2 is a descendant of 1
    assert rels[2, 3] == Models.diff_branches  # siblings
    assert rels[2, 4] == Models.diff_branches  # different subtrees
    assert np.all(np.diag(rels) == Models.cocluster)


# ---- Tests on many trees ----
#
# The tests above check one hand-drawn tree. The tests below run the same kinds
# of checks on many trees: some chosen to be extreme shapes, and many random ones.

def random_parents(n_nodes, seed):
    # Builds a random tree by adding non-root nodes one at a time, in random
    # order, each attached to a random node already in the tree. Because nodes
    # are added in random order, a node's parent can have a higher index than
    # the node itself, as can happen in trees produced by MCMC.
    rng = np.random.default_rng(seed)
    parents = np.zeros(n_nodes - 1, dtype=np.int64)
    in_tree = [0]
    for node in rng.permutation(np.arange(1, n_nodes)):
        parents[node - 1] = rng.choice(in_tree)
        in_tree.append(node)
    return parents


def reference_ancmatrix(parents):
    # A deliberately simple (and slow) way to build the ancestry matrix: for each
    # node, walk up through its parents to the root, marking each one as an
    # ancestor. Written independently of tree_util so the two can be compared.
    n_nodes = len(parents) + 1
    anc = np.eye(n_nodes, dtype=int)
    for node in range(1, n_nodes):
        ancestor = node
        while ancestor != 0:
            ancestor = parents[ancestor - 1]
            anc[ancestor, node] = 1
    return anc


# Each pytest.param is one tree; `id` is the name pytest shows for it.
TREES = [
    pytest.param(np.array([0]), id="single-child"),
    pytest.param(np.arange(9), id="chain"),             # 0 -> 1 -> 2 -> ... -> 9
    pytest.param(np.zeros(9, dtype=np.int64), id="star"),  # every node a child of 0
    pytest.param(np.array([2, 0, 1]), id="chain-unordered"),  # 0 -> 2 -> 3 -> 1
] + [
    pytest.param(random_parents(n_nodes, seed), id=f"random-n{n_nodes}-seed{seed}")
    for n_nodes in (3, 10, 50)
    for seed in range(10)
]


# @pytest.mark.parametrize runs the test once for every entry in TREES, passing
# that entry in as `parents`. Each run is reported as a separate test.
@pytest.mark.parametrize("parents", TREES)
def test_ancmatrix_matches_reference(parents):
    anc = tree_util.convert_parents_to_ancmatrix(parents)
    assert np.array_equal(anc, reference_ancmatrix(parents))


@pytest.mark.parametrize("parents", TREES)
def test_parents_adjmatrix_roundtrip(parents):
    adj = tree_util.convert_parents_to_adjmatrix(parents)
    assert np.array_equal(tree_util.convert_adjmatrix_to_parents(adj), parents)


@pytest.mark.parametrize("parents", TREES)
def test_parents_ancmatrix_roundtrip(parents):
    anc = tree_util.convert_parents_to_ancmatrix(parents)
    assert np.array_equal(tree_util.convert_ancmatrix_to_parents(anc), parents)


@pytest.mark.parametrize("parents", TREES)
def test_adjmatrix_ancmatrix_roundtrip(parents):
    adj = tree_util.convert_parents_to_adjmatrix(parents)
    anc = tree_util.convert_adjmatrix_to_ancmatrix(adj)
    assert np.array_equal(tree_util.convert_ancmatrix_to_adjmatrix(anc), adj)


@pytest.mark.parametrize("parents", TREES)
def test_node_relations_match_ancmatrix(parents):
    adj = tree_util.convert_parents_to_adjmatrix(parents)
    rels = tree_util.compute_node_relations(adj)
    anc = reference_ancmatrix(parents)

    n_nodes = len(parents) + 1
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                expected = Models.cocluster
            elif anc[i, j]:
                expected = Models.A_B
            elif anc[j, i]:
                expected = Models.B_A
            else:
                expected = Models.diff_branches
            assert rels[i, j] == expected, f"wrong relation for nodes {i}, {j}"
