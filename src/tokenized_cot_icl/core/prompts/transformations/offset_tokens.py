"""Offset the token id's so as to train NLP models"""

def offset_tokens_fn(batch, offset: int):
    for item in batch:
        item["input_ids"] += offset
        item["labels"] += offset
        item["cot_eval"]["input_ids"] += offset
        item["cot_eval"]["last_example_cot"] += offset
    return batch