import random
import torch
from tokenized_cot_icl.core.prompts.transformations.utils import get_cot_tokens

def _drop_first_k_cot_tokens(tokens, start_token, end_token, k):
    tokens, cots = get_cot_tokens(tokens, start_token, end_token, k)
    if cots is None:
        return tokens
    cots = cots[:k]
    keep_mask = torch.ones_like(tokens, dtype=torch.bool)

    for start_pos, end_pos in cots:
        keep_mask[start_pos: end_pos + 1] = False

    return tokens[keep_mask]

def _drop_last_k_cot_tokens(tokens, start_token, end_token, k):
    tokens, cots = get_cot_tokens(tokens, start_token, end_token, k)
    if cots is None:
        return tokens
    cots = cots[-k:]
    keep_mask = torch.ones_like(tokens, dtype=torch.bool)

    for start_pos, end_pos in cots:
        keep_mask[start_pos: end_pos + 1] = False

    return tokens[keep_mask]

def _drop_random_k_cot_tokens(tokens, start_token, end_token, k):
    tokens, cots = get_cot_tokens(tokens, start_token, end_token, k)
    if cots is None:
        return tokens
    keep_mask = torch.ones_like(tokens, dtype=torch.bool)
    cots = random.sample(cots, k=k)

    for start_pos, end_pos in cots:
        keep_mask[start_pos: end_pos + 1] = False

    return tokens[keep_mask]

COT_DROPPING_FUNCTIONS = {
    "first_k": _drop_first_k_cot_tokens,
    "last_k": _drop_last_k_cot_tokens,
    "random": _drop_random_k_cot_tokens
}

def drop_k_cot_tokens_fn(batch, transformation: str,  k: int):
    think_start_token_id = batch[0]["think_start_token_id"]
    think_end_token_id = batch[0]["think_end_token_id"]

    # drop token ids from "cot_eval" prompt in the batch
    drop_fn = COT_DROPPING_FUNCTIONS[transformation]
    for item in batch:
        token_ids = item["cot_eval"]["input_ids"]
        token_ids = drop_fn(token_ids, think_start_token_id, think_end_token_id, k)
        item["cot_eval"]["input_ids"] = token_ids
        item["cot_eval"]["attention_mask"] = torch.ones(len(token_ids))

    return batch

