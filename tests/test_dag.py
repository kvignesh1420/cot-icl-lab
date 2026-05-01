import numpy as np
import pytest

from tokenized_cot_icl.core.data.dag import DAG_REGISTRY


@pytest.mark.parametrize(
    "n_inputs, n_parents, chain_length",
    [
        (4, 1, 4),
        (4, 2, 4),
        (4, 3, 4),
        (4, 4, 4),
        (8, 2, 6),
        (1, 1, 5),
    ],
)
def test_dag_shape(n_inputs, n_parents, chain_length):
    dag = DAG_REGISTRY["random"](n_inputs, n_parents, chain_length)
    adj_list = dag.generate_adj_list()
    assert len(adj_list) == chain_length
    for parents in adj_list:
        assert len(parents) == n_parents


@pytest.mark.parametrize(
    "n_inputs, n_parents, chain_length",
    [
        (4, 2, 4),
        (8, 3, 6),
        (5, 1, 7),
    ],
)
def test_dag_parents_are_acyclic(n_inputs, n_parents, chain_length):
    """Each parent index must reference a node strictly earlier in topological order."""
    np.random.seed(0)
    dag = DAG_REGISTRY["random"](n_inputs, n_parents, chain_length)
    adj_list = dag.generate_adj_list()
    for chain_idx, parents in enumerate(adj_list):
        # chain token at position (n_inputs + chain_idx) may only depend on
        # indices < n_inputs + chain_idx
        max_valid_index = n_inputs + chain_idx - 1
        for p in parents:
            assert 0 <= p <= max_valid_index, (
                f"chain_idx={chain_idx} parent={p} exceeds max valid index {max_valid_index}"
            )


@pytest.mark.parametrize(
    "n_inputs, n_parents, chain_length",
    [
        (4, 4, 3),
        (8, 3, 5),
    ],
)
def test_dag_parents_unique_within_step(n_inputs, n_parents, chain_length):
    np.random.seed(0)
    dag = DAG_REGISTRY["random"](n_inputs, n_parents, chain_length)
    adj_list = dag.generate_adj_list()
    for parents in adj_list:
        assert len(set(parents)) == len(parents)
