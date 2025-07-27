from typing import Optional
import numpy as np
import torch
from torch import nn

from tokenized_cot_icl.core.utils import set_random_seed


def get_activation_fn(activation: str):
    if activation == "identity":
        return nn.Identity()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(0.5)
    else:
        raise ValueError(f"Activation function {activation} not supported.")


class TokenProcessor(nn.Module):
    """Given an embedding of a token, process it via Fully Connected Networks"""

    def __init__(self, n_dims: int, num_layers: int, activation: str):
        super().__init__()
        self.n_dims = n_dims
        self.num_layers = num_layers
        self.activation_fn = get_activation_fn(activation=activation)
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.n_dims, self.n_dims, bias=False)
                for _ in range(self.num_layers)
            ]
        )
        for layer in self.layers:
            torch.nn.init.normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation to all hidden layers except the last"""
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)


class TokenProcessorCache:
    """A possibly infinite collection of TokenProcessors

    Args:
        maxsize: The size of the list of processors to uniformly sample from.
            Use `None` for an infinite stream.
    """

    def __init__(
        self, maxsize: Optional[int], n_dims: int, num_layers: int, activation: str
    ):
        self.maxsize = maxsize if maxsize else np.inf
        self.n_dims = n_dims
        self.num_layers = num_layers
        self.activation = activation
        self.storage = []

    def add_to_storage(self, token_processor: TokenProcessor):
        if np.isinf(self.maxsize):
            return
        self.storage.append(token_processor)

    def sample(self):
        assert (
            len(self.storage) <= self.maxsize
        ), f"storage size: {len(self.storage)} has exceeded maxsize: {self.maxsize}."
        if len(self.storage) < self.maxsize:
            token_processor = TokenProcessor(
                n_dims=self.n_dims,
                num_layers=self.num_layers,
                activation=self.activation,
            )
            self.add_to_storage(token_processor=token_processor)
        else:
            idx = np.random.randint(self.maxsize)
            token_processor = self.storage[idx]
        return token_processor

    def warmup(self, seed: int):
        set_random_seed(seed)
        if np.isinf(self.maxsize):
            return
        for _ in range(self.maxsize):
            self.sample()
