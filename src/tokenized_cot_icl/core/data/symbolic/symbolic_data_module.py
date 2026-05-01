from copy import deepcopy
from typing import Dict, List

import numpy as np
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer

from tokenized_cot_icl.core.data.recipes import RECIPE_REGISTRY
from tokenized_cot_icl.core.data.symbolic.common import ANSWER_PREFIX, SYSTEM_PROMPT, THINK_PREFIX
from tokenized_cot_icl.core.utils import set_random_seed


class SymbolicDataModule:
    """
    DataModule for the the SymbolicDataModule datasets.

    ```py
    >>> dm = SymbolicDataModule(
        dataset_path="/path/to/cot_icl_lab_symbolic_N_4_M_2_C_4_K_40_word_len_8",
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"),
        examples_per_prompt=2,
        cot_example_prob_recipe_info={
            "type": "linear",
            "initial_prob": 1.0,
            "final_prob": 1.0,
        },
    )
    >>> dm.process()
    >>> print(dm.train_dataset[0]["text"])
    ```

    """

    def __init__(
        self,
        dataset_path: str,
        tokenizer: AutoTokenizer,
        examples_per_prompt: int,
        cot_example_prob_recipe_info: Dict,
        seed: int = 42,
        **kwargs,
    ):
        self.dataset: Dataset = load_from_disk(dataset_path)
        self.column_names = self.dataset["train"].column_names
        self.tokenizer: AutoTokenizer = tokenizer
        self.examples_per_prompt = examples_per_prompt
        self.cot_example_prob_recipe_info = cot_example_prob_recipe_info
        self.seed = seed

    def setup_recipe(self, n_prompts: int):
        recipe_clz = RECIPE_REGISTRY[self.cot_example_prob_recipe_info["type"]]
        recipe_kwargs = deepcopy(self.cot_example_prob_recipe_info)
        recipe_kwargs["n_prompts"] = n_prompts
        self.cot_example_prob_recipe = recipe_clz(**recipe_kwargs)

    def prepare_question(self, input_words):
        return (
            "Given the input words: "
            + ", ".join(input_words)
            + ", what is the final answer after the string transformations?"
        )

    def prepare_cot_solution(self, intermediate_words: List[str], answer_word: str):
        reasoning = THINK_PREFIX
        for i in range(len(intermediate_words)):
            reasoning += f"Step {i + 1}: {intermediate_words[i]}\n"
        reasoning += f"{ANSWER_PREFIX}\\boxed{{{answer_word}}}"
        return reasoning

    def prepare_direct_solution(self, answer_word: str):
        return f"{ANSWER_PREFIX}\\boxed{{{answer_word}}}"

    def prepare_prompt(self, row, prompt_index: int) -> Dict:
        cot_example_prob = self.cot_example_prob_recipe.get_value(prompt_index=prompt_index)

        conversation = [{"role": "system", "content": SYSTEM_PROMPT}]
        examples = row["examples"][: self.examples_per_prompt]

        for i in range(len(examples)):
            question = self.prepare_question(input_words=examples[i]["input_words"])
            cot_solution_with_prefix = self.prepare_cot_solution(
                intermediate_words=examples[i]["intermediate_words"],
                answer_word=examples[i]["answer_word"],
            )
            answer_with_prefix = self.prepare_direct_solution(answer_word=examples[i]["answer_word"])
            conversation.extend(
                [
                    {
                        "role": "user",
                        "content": question,
                    },
                    {
                        "role": "assistant",
                        "content": (
                            cot_solution_with_prefix if np.random.rand() < cot_example_prob else answer_with_prefix
                        ),
                    },
                ]
            )

        prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        return {
            "text": prompt,
            "cot_example_prob": cot_example_prob,
        }

    def process(self):
        """
        Process the dataset into a chat format.
        """
        set_random_seed(seed=self.seed)
        train_dataset = self.dataset["train"]
        print(f"Original train dataset size: {len(train_dataset)}")
        # after batching based on examples_per_prompt
        n_prompts = len(train_dataset)
        print(f"Setting up the recipe for {n_prompts} prompts.")
        self.setup_recipe(n_prompts=n_prompts)

        train_dataset = train_dataset.map(
            self.prepare_prompt,
            with_indices=True,
            batched=False,
            remove_columns=self.column_names,
        )

        self.train_dataset = train_dataset.shuffle(seed=self.seed)
        self.eval_dataset = None
