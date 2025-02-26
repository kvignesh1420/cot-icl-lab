"""A registry of all prompts."""

from tokenized_cot_icl.core.prompts.cot import CoTPrompt
from tokenized_cot_icl.core.prompts.standard import StandardPrompt

PROMPT_REGISTRY = {
    "standard": StandardPrompt,
    "cot": CoTPrompt,
}
