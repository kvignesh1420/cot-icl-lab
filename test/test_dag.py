import pytest
from tokenized_cot_icl.core.dag import DAG_REGISTRY


@pytest.mark.parametrize(
    "n_inputs, n_parents, chain_length",
    [
        (4, 1, 4),
        (4, 2, 4),
        (4, 3, 4),
        (4, 4, 4),
    ],
)
def test_dag(n_inputs, n_parents, chain_length):
    dag = DAG_REGISTRY["random"](n_inputs, n_parents, chain_length)
    adj_list = dag.generate_adj_list()
    assert len(adj_list) == chain_length
    assert len(adj_list[0]) == n_parents
