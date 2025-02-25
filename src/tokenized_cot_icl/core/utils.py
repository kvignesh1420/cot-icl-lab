"""Common utilities"""

import logging

logging.basicConfig(level=logging.INFO)

from datetime import datetime
import random
import numpy as np

from tokenized_cot_icl.core.args import Args, MLFLOW_SERVICE_URL


def set_random_seed(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_mlflow_client():
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_SERVICE_URL)
    return mlflow.client.MlflowClient()


def get_mlflow_run(args: Args):
    import mlflow

    mlflow_client = get_mlflow_client()
    experiment_name = f"cot-icl-lab-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    experiment = mlflow.set_experiment(experiment_name)
    run_name = prepare_run_name(args)
    run = mlflow_client.create_run(
        experiment_id=experiment.experiment_id, run_name=run_name
    )
    return run


def prepare_run_name(args: Args) -> str:
    """Prepare the run name for the experiment based on the arguments.
    Example : llama_L_4_V_16384_H_8_M_4_N_4_C_2_n_ex_40_n_dims_10_hs_256_bs_128_act_leaky_relu_data_std_1_cot_false
    """
    substrings = [
        args.model_type,
        f"L_{args.num_hidden_layers}",
        f"V_{args.vocab_size}",
        f"H_{args.num_attention_heads}",
        f"M_{args.n_parents}",
        f"N_{args.n_inputs}",
        f"C_{args.chain_length}",
        f"n_ex_{args.n_examples}",
        f"n_dims_{args.n_dims}",
        f"hs_{args.hidden_size}",
        f"bs_{args.batch_size}",
        f"act_{args.activation}",
        f"data_std_{args.data_initializer_range}",
        f"cot_{args.enable_cot}",
    ]
    return "_".join(substrings)
