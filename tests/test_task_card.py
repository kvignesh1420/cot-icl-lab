import pytest

from tokenized_cot_icl.core.task_card import (
    baseline_fixed_chain_length,
    hybrid_power_law_alpha,
    probabilistic_chain_length,
    vary_chain_length,
    vary_data_activation,
    vary_n_dims,
    vary_n_parents,
    vary_num_examples,
    vary_vocab_size,
)


@pytest.mark.parametrize(
    "create_fn",
    [
        vary_vocab_size,
        vary_num_examples,
        vary_chain_length,
        vary_data_activation,
        vary_n_dims,
        vary_n_parents,
        baseline_fixed_chain_length,
        probabilistic_chain_length,
        hybrid_power_law_alpha,
    ],
)
def test_task_card_init(create_fn):
    assert create_fn()
