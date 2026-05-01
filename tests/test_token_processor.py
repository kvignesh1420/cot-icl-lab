import pytest
import torch
from torch import nn

from tokenized_cot_icl.core.args import Args
from tokenized_cot_icl.core.data.token_processor import TokenProcessor, TokenProcessorCache, get_activation_fn


def test_token_processor_init():
    args = Args()
    token_processor = TokenProcessor(n_dims=args.n_dims, num_layers=args.H_num_layers, activation=args.activation)
    assert token_processor.num_layers == args.H_num_layers
    assert len(token_processor.layers) == args.H_num_layers


@pytest.mark.parametrize(
    "activation, expected_cls",
    [
        ("identity", nn.Identity),
        ("relu", nn.ReLU),
        ("silu", nn.SiLU),
        ("leaky_relu", nn.LeakyReLU),
    ],
)
def test_get_activation_fn(activation, expected_cls):
    assert isinstance(get_activation_fn(activation=activation), expected_cls)


def test_get_activation_fn_unknown_raises():
    with pytest.raises(ValueError):
        get_activation_fn(activation="not-a-real-activation")


@pytest.mark.parametrize("num_layers", [1, 2, 3])
def test_token_processor_forward_shape(num_layers):
    n_dims = 8
    tp = TokenProcessor(n_dims=n_dims, num_layers=num_layers, activation="leaky_relu")
    x = torch.randn(5, n_dims)
    y = tp(x)
    assert y.shape == (5, n_dims)


def test_token_processor_cache_infinite_does_not_store():
    cache = TokenProcessorCache(maxsize=None, n_dims=4, num_layers=1, activation="relu")
    for _ in range(5):
        sampled = cache.sample()
        assert isinstance(sampled, TokenProcessor)
    # infinite cache never accumulates storage
    assert cache.storage == []


def test_token_processor_cache_finite_caps_storage():
    cache = TokenProcessorCache(maxsize=3, n_dims=4, num_layers=1, activation="relu")
    seen = [cache.sample() for _ in range(3)]
    assert len(cache.storage) == 3
    # further samples must come from the existing pool
    for _ in range(10):
        sampled = cache.sample()
        assert sampled in seen
    assert len(cache.storage) == 3


def test_token_processor_cache_warmup_is_deterministic():
    cache_a = TokenProcessorCache(maxsize=4, n_dims=4, num_layers=1, activation="relu")
    cache_b = TokenProcessorCache(maxsize=4, n_dims=4, num_layers=1, activation="relu")
    cache_a.warmup(seed=123)
    cache_b.warmup(seed=123)
    assert len(cache_a.storage) == len(cache_b.storage) == 4
    for tp_a, tp_b in zip(cache_a.storage, cache_b.storage):
        for layer_a, layer_b in zip(tp_a.layers, tp_b.layers):
            assert torch.allclose(layer_a.weight, layer_b.weight)


def test_token_processor_cache_warmup_no_op_for_infinite():
    cache = TokenProcessorCache(maxsize=None, n_dims=4, num_layers=1, activation="relu")
    cache.warmup(seed=0)
    assert cache.storage == []
