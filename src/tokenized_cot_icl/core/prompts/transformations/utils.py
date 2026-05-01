def get_cot_tokens(tokens, start_token, end_token, k):
    start_positions = (tokens == start_token).nonzero(as_tuple=True)[0]
    end_positions = (tokens == end_token).nonzero(as_tuple=True)[0]

    if len(start_positions) == 0 or len(end_positions) == 0 or k == 0:
        return tokens, None

    assert len(start_positions) == len(end_positions), (
        f"Invalid placement of think special tokens. The number of start think and end think do not match {len(start_positions)} vs {len(end_positions)}."
    )
    cots = [(start, end) for start, end in zip(start_positions, end_positions)]
    assert k <= len(cots), (
        f"The number of CoT examples {len(cots)} is less than the number of CoT examples to drop {k}. Please check your input."
    )
    return tokens, cots
