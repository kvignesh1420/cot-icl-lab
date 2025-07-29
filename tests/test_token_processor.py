from tokenized_cot_icl.core.args import Args
from tokenized_cot_icl.core.data.token_processor import TokenProcessor


def test_token_processor():
    args = Args()
    token_processor = TokenProcessor(n_dims=args.n_dims, num_layers=args.H_num_layers, activation=args.activation)
    assert token_processor.num_layers == args.H_num_layers
    assert len(token_processor.layers) == args.H_num_layers
