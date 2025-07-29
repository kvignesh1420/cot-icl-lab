"""Common utilities"""

import logging
import random
from datetime import datetime
from typing import Union

import numpy as np

from tokenized_cot_icl.core.args import Args, EvalArgs
from tokenized_cot_icl.core.metric_loggers import METRIC_LOGGER_REGISTRY, MetricLogger

logging.basicConfig(level=logging.INFO)


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


def prepare_run_name(args: Union[Args, EvalArgs]) -> str:
    run_name = f"{str(args)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return run_name


def create_metric_logger(args: Union[Args, EvalArgs], device_id=0):
    assert args.metric_logger in METRIC_LOGGER_REGISTRY, f"Metric logger {args.metric_logger} not supported."
    run_name = prepare_run_name(args)
    metric_logger: MetricLogger = METRIC_LOGGER_REGISTRY[args.metric_logger](run_name=run_name, device_id=device_id)
    metric_logger.log_params(params=args.__dict__)
    return metric_logger
