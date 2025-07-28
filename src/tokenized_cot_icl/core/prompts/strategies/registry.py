"""A registry of all prompts.strategies."""

from tokenized_cot_icl.core.prompts.strategies.standard import StandardPrompt
from tokenized_cot_icl.core.prompts.strategies.standard_special_token import (
    StandardSpecialTokenPrompt,
)
from tokenized_cot_icl.core.prompts.strategies.cot import CoTPrompt
from tokenized_cot_icl.core.prompts.strategies.cot_special_token import CoTSpecialTokenPrompt
from tokenized_cot_icl.core.prompts.strategies.hybrid_special_token import HybridSpecialTokenPrompt

PROMPT_STRATEGY_REGISTRY = {
    "standard": StandardPrompt,
    "cot": CoTPrompt,
    "standard_special_token": StandardSpecialTokenPrompt,
    "cot_special_token": CoTSpecialTokenPrompt,
    "hybrid_special_token": HybridSpecialTokenPrompt,
}
