"""A registry of all prompts."""

from tokenized_cot_icl.core.prompts.standard import StandardPrompt
from tokenized_cot_icl.core.prompts.cot import CoTPrompt

PROMPT_REGISTRY = {
    "standard": StandardPrompt,
    "cot": CoTPrompt,
}
