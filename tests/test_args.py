import pytest

from tokenized_cot_icl.core.args import Args


def test_args():
    args = Args(enable_special_tokens=False)
    assert args.pad_token_id is None
    assert args.bos_token_id == 2 * args.vocab_size + 2
    assert args.eos_token_id == 2 * args.vocab_size + 3
    assert args.input_start_token_id == 2 * args.vocab_size + 4
    assert args.input_end_token_id == 2 * args.vocab_size + 5
    assert args.think_start_token_id == 2 * args.vocab_size + 6
    assert args.think_end_token_id == 2 * args.vocab_size + 7
    assert args.answer_start_token_id == 2 * args.vocab_size + 8
    assert args.answer_end_token_id == 2 * args.vocab_size + 9
    assert args.reserved_token_ids == {}
    assert (
        str(args)
        == "llama_L_4_V_1024_heads_32_M_4_N_4_C_2_n_ex_40_n_dims_10_hs_2048_bs_64_act_leaky_relu_prompt_standard_recipe_none"
    )


def test_args_with_choices():
    args = Args(
        n_parent_choices=(1, 2),
        n_input_choices=(3, 4),
        chain_length_choices=(5, 6),
        n_example_choices=(7, 8),
        enable_special_tokens=False,
    )
    assert (
        str(args)
        == "llama_L_4_V_1024_heads_32_M_1-2_N_3-4_C_5-6_n_ex_7-8_n_dims_10_hs_2048_bs_64_act_leaky_relu_prompt_standard_recipe_none"
    )


def test_args_with_choices_and_recipe():
    args = Args(
        n_parent_choices=(1, 2),
        n_input_choices=(3, 4),
        chain_length_choices=(5, 6),
        n_example_choices=(7, 8),
        enable_special_tokens=False,
        cot_example_prob_recipe_info={"type": "power_law", "alpha": 0.5},
    )
    assert (
        str(args)
        == "llama_L_4_V_1024_heads_32_M_1-2_N_3-4_C_5-6_n_ex_7-8_n_dims_10_hs_2048_bs_64_act_leaky_relu_prompt_standard_recipe_type_power_law_alpha_0.5"
    )


@pytest.mark.parametrize("vocab_size", [10, 20, 30])
def test_special_token_args(vocab_size):
    args = Args(vocab_size=vocab_size, enable_special_tokens=True)
    assert args.pad_token_id == vocab_size - 1
    assert args.bos_token_id == vocab_size - 2
    assert args.eos_token_id == vocab_size - 3
    assert args.input_start_token_id == vocab_size - 4
    assert args.input_end_token_id == vocab_size - 5
    assert args.think_start_token_id == vocab_size - 6
    assert args.think_end_token_id == vocab_size - 7
    assert args.answer_start_token_id == vocab_size - 8
    assert args.answer_end_token_id == vocab_size - 9
    assert list(args.reserved_token_ids.values()) == [
        vocab_size - 1,
        vocab_size - 2,
        vocab_size - 3,
        vocab_size - 4,
        vocab_size - 5,
        vocab_size - 6,
        vocab_size - 7,
        vocab_size - 8,
        vocab_size - 9,
    ]
