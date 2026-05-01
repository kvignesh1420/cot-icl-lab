import argparse
import os
import random
import string
from copy import deepcopy
from typing import Dict, List

import datasets
from tqdm import tqdm

from tokenized_cot_icl.core.data.dag import RandomDAG
from tokenized_cot_icl.core.utils import set_random_seed

CHARS = list(string.ascii_lowercase)


def create_word(length: int) -> str:
    """
    Create a random word of a given length using uppercase letters.
    """
    return "".join(random.choice(CHARS) for _ in range(length))


def split_and_concat_words(words: List[str]) -> str:
    """Take the second half of each word and concatenate them."""
    half_words = [word[len(word) // 2 :] for word in words]
    return "".join(half_words)


def offset_characters(word: str, offset: int) -> str:
    """
    Offset each character in the word by a given offset.
    """
    return "".join(CHARS[(CHARS.index(c) + offset) % len(CHARS)] for c in word)


def create_icl_example(word_length: int, num_words: int, adj_list: list, char_offset: int) -> Dict[str, str]:
    """
    Create an ICL example by generating random words and concatenating their second halves.
    """
    input_words = [create_word(length=word_length) for _ in range(num_words)]
    all_words = deepcopy(input_words)
    chain_words = []
    for parents in adj_list:
        words = [all_words[idx] for idx in parents]
        chain_word = split_and_concat_words(words=words)
        chain_word = offset_characters(word=chain_word, offset=char_offset)
        chain_words.append(chain_word)
        all_words.append(chain_word)

    return {
        "input_words": input_words,
        "intermediate_words": chain_words[:-1],
        "answer_word": chain_words[-1],
        "adj_list": adj_list,
    }


def get_icl_examples(
    word_length: int,
    n_inputs: int,
    n_parents: int,
    chain_length: int,
    num_examples: int,
    char_offset: int,
) -> List[Dict[str, str]]:
    dag = RandomDAG(n_inputs=n_inputs, n_parents=n_parents, chain_length=chain_length)
    adj_list = dag.generate_adj_list()
    adj_list = [sorted(parents) for parents in adj_list]
    icl_examples = [
        create_icl_example(
            word_length=word_length,
            num_words=n_inputs,
            adj_list=adj_list,
            char_offset=char_offset,
        )
        for _ in range(num_examples)
    ]
    return icl_examples


def prepare_dataset(
    num_examples: int,
    word_length: int,
    n_inputs: int,
    n_parents: int,
    chain_length: int,
    char_offset: int,
    n_tasks: int,
    n_eval_tasks: int,
    output_dir: str,
) -> List[Dict[str, str]]:
    """
    Prepare a dataset of ICL examples.
    """
    train_data = []
    for _ in tqdm(range(n_tasks)):
        train_data.append(
            {
                "examples": get_icl_examples(
                    word_length=word_length,
                    n_inputs=n_inputs,
                    n_parents=n_parents,
                    chain_length=chain_length,
                    num_examples=num_examples,
                    char_offset=char_offset,
                )
            }
        )
    train_ds = datasets.Dataset.from_list(train_data)

    test_data = []
    for _ in tqdm(range(n_eval_tasks)):
        test_data.append(
            {
                "examples": get_icl_examples(
                    word_length=word_length,
                    n_inputs=n_inputs,
                    n_parents=n_parents,
                    chain_length=chain_length,
                    num_examples=num_examples,
                    char_offset=char_offset,
                )
            }
        )

    test_ds = datasets.Dataset.from_list(test_data)

    dataset = datasets.DatasetDict({"train": train_ds, "test": test_ds})
    print(dataset)

    save_path = os.path.join(
        output_dir,
        f"cot_icl_lab_symbolic_N_{n_inputs}_M_{n_parents}_C_{chain_length}_K_{num_examples}_word_len_{word_length}_char_offset_{char_offset}_n_tasks_{n_tasks}_n_eval_tasks_{n_eval_tasks}",
    )
    dataset.save_to_disk(save_path)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for ICL examples.")
    parser.add_argument("--num_examples", type=int, default=40, help="Number of examples per task.")
    parser.add_argument("--word_length", type=int, default=8, help="Length of the words.")
    parser.add_argument("--n_inputs", type=int, default=4, help="Number of input words.")
    parser.add_argument("--n_parents", type=int, default=2, help="Number of parents for each word.")
    parser.add_argument("--chain_length", type=int, default=3, help="Length of the chain.")
    parser.add_argument(
        "--char_offset",
        type=int,
        default=1,
        help="Offset for the characters in the words.",
    )
    parser.add_argument("--n_tasks", type=int, default=1000, help="Number of tasks.")
    parser.add_argument("--n_eval_tasks", type=int, default=10000, help="Number of evaluation tasks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory in which to save the prepared dataset.",
    )
    args = parser.parse_args()
    print(f"prepare dataset with args: {args}")

    # Set the random seed for reproducibility
    set_random_seed(args.seed)

    prepare_dataset(
        num_examples=args.num_examples,
        word_length=args.word_length,
        n_inputs=args.n_inputs,
        n_parents=args.n_parents,
        chain_length=args.chain_length,
        char_offset=args.char_offset,
        n_tasks=args.n_tasks,
        n_eval_tasks=args.n_eval_tasks,
        output_dir=args.output_dir,
    )
