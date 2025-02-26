import torch
from torch import nn

from tokenized_cot_icl.core.args import Args


def get_activation_fn(args: Args):
    if args.activation == "identity":
        return nn.Identity()
    elif args.activation == "silu":
        return nn.SiLU()
    elif args.activation == "relu":
        return nn.ReLU()
    elif args.activation == "leaky_relu":
        return nn.LeakyReLU(0.5)
    else:
        raise ValueError(f"Activation function {args.activation} not supported.")


class TokenProcessor(nn.Module):
    """Given an embedding of a token, process it via Fully Connected Networks"""

    def __init__(self, args: Args):
        super().__init__()
        self.args = args
        self.activation_fn = get_activation_fn(args=self.args)
        self.num_layers = self.args.H_num_layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.args.n_dims, self.args.n_dims, bias=False) for _ in range(self.num_layers)]
        )
        for layer in self.layers:
            torch.nn.init.normal_(layer.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation to all hidden layers except the last"""
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)
