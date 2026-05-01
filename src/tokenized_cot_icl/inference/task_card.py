import os
from itertools import product
from typing import Dict, List

from tokenized_cot_icl.core.args import EvalArgs


def _get_paths(model_info: Dict) -> Dict[str, str]:
    """Get paths for model and output directory."""
    model_path = os.path.join(model_info["base_path"], model_info["model_checkpoint"])
    output_dir = os.path.join(model_info["base_path"], model_info["eval_path"])
    return {"model_path": model_path, "output_dir": output_dir}


def default_task_card(model_infos: List[Dict]) -> Dict[int, EvalArgs]:
    """Default task card."""
    args_dict = {}
    for idx, model_info in enumerate(model_infos):
        args = EvalArgs(
            **_get_paths(model_info),
        )
        args_dict[idx] = args
    return args_dict


def vary_forced_special_tokens(model_infos: List[Dict]) -> Dict[int, EvalArgs]:
    """Switch between forced and non-forced special tokens."""

    args_dict = {}
    ft_fa_pairs = [
        (False, False),  # free form generation
        (True, False),  # force think token
        (False, True),  # force answer token
    ]
    settings = product(ft_fa_pairs, model_infos)
    for idx, ((ft, fa), model_info) in enumerate(settings):
        args = EvalArgs(**_get_paths(model_info), force_think_token=ft, force_answer_token=fa, n_eval_tasks=10000)
        args_dict[idx] = args
    return args_dict


def vary_input_and_chain_len(model_infos: List[Dict]) -> Dict[int, EvalArgs]:
    """Vary input length and chain length."""
    input_lens = [4]
    chain_lens = [5]
    ft_fa_pairs = [
        (False, False),  # free form generation
        (True, False),  # force think token
        (False, True),  # force answer token
    ]
    settings = product(input_lens, chain_lens, ft_fa_pairs, model_infos)
    args_dict = {}

    for idx, (n_input, chain_len, (ft, fa), model_info) in enumerate(settings):
        args = EvalArgs(
            **_get_paths(model_info),
            force_think_token=ft,
            force_answer_token=fa,
            n_input=n_input,
            chain_len=chain_len,
            n_eval_tasks=10000,
        )
        args_dict[idx] = args
    return args_dict


def vary_temperature_and_output_seqs(model_infos: List[Dict]) -> Dict[int, EvalArgs]:
    """Vary temperature and number of output sequences."""
    temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]
    num_output_seqs = [4, 8, 16]
    ft_fa_pairs = [
        # (False, False),  # free form generation
        (True, False),  # force think token
        # (False, True),  # force answer token
    ]
    settings = product(temperatures, num_output_seqs, ft_fa_pairs, model_infos)

    args_dict = {}
    for idx, (temp, num_seqs, (ft, fa), model_info) in enumerate(settings):
        args = EvalArgs(
            **_get_paths(model_info),
            force_think_token=ft,
            force_answer_token=fa,
            temperature=temp,
            num_output_seqs=num_seqs,
            n_eval_tasks=10000,
        )
        args_dict[idx] = args
    return args_dict


def vary_n_examples(model_infos: List[Dict]) -> Dict[int, EvalArgs]:
    """Vary the number of examples in the dataset."""
    ft_fa_pairs = [
        (False, False),  # free form generation
        (True, False),  # force think token
        (False, True),  # force answer token
    ]
    num_examples = [10, 20, 30, 40, 50, 60]
    settings = product(num_examples, ft_fa_pairs, model_infos)
    args_dict = {}
    for idx, (n_examples, (ft, fa), model_info) in enumerate(settings):
        args = EvalArgs(
            **_get_paths(model_info),
            force_think_token=ft,
            force_answer_token=fa,
            n_examples=n_examples,
        )
        args_dict[idx] = args
    return args_dict


####### Model Info #######

BASE_PATH = "/opt/cot-icl-lab/"
MODEL_INFOS = [
    {
        "base_path": os.path.join(
            BASE_PATH,
            f"llama_L_{L}_V_1024_heads_32_M_4_N_4_C_2_n_ex_40_n_dims_10_hs_2048_bs_32_act_leaky_relu_prompt_hybrid_special_token_recipe_type_power_law_alpha_{alpha}",
        ),
        "model_checkpoint": "final_model",
        "eval_path": "custom_evals",
    }
    for L in [4]
    for alpha in ["2"]
]


EVAL_TASK_CARD = default_task_card(model_infos=MODEL_INFOS)
# EVAL_TASK_CARD = vary_temperature_and_output_seqs(model_infos=MODEL_INFOS)
# EVAL_TASK_CARD = vary_n_examples(model_infos=MODEL_INFOS)

# EVAL_TASK_CARD = vary_forced_special_tokens(model_infos=MODEL_INFOS)
# EVAL_TASK_CARD = vary_input_and_chain_len(model_infos=MODEL_INFOS)
