from tokenized_cot_icl.core.prompts.transformations.drop_cot_tokens import (
    drop_k_cot_tokens_fn,
)
from tokenized_cot_icl.core.prompts.transformations.offset_tokens import offset_tokens_fn

PROMPT_TRANSFORMATION_REGISTRY = {
    "drop_k_cot_tokens": drop_k_cot_tokens_fn,
    "offset_tokens": offset_tokens_fn
}