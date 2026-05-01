import os
import string

import pytest

from tokenized_cot_icl.core.data.symbolic.prepare import (
    CHARS,
    create_icl_example,
    create_word,
    get_icl_examples,
    offset_characters,
    prepare_dataset,
    split_and_concat_words,
)
from tokenized_cot_icl.core.utils import set_random_seed


@pytest.mark.parametrize("length", [1, 4, 8, 16])
def test_create_word_length_and_alphabet(length):
    set_random_seed(0)
    word = create_word(length=length)
    assert len(word) == length
    assert all(c in string.ascii_lowercase for c in word)


def test_split_and_concat_words_takes_second_half():
    # second halves: "cd", "gh", "kl"
    result = split_and_concat_words(["abcd", "efgh", "ijkl"])
    assert result == "cdghkl"


def test_split_and_concat_words_odd_length_uses_floor_split():
    # len("abcde") // 2 == 2 -> second half is "cde"
    result = split_and_concat_words(["abcde"])
    assert result == "cde"


@pytest.mark.parametrize(
    "word, offset, expected",
    [
        ("abc", 1, "bcd"),
        ("xyz", 1, "yza"),  # wrap-around
        ("abc", 0, "abc"),  # identity
        ("abc", 26, "abc"),  # full cycle
        ("abc", 27, "bcd"),  # wrap > 26
    ],
)
def test_offset_characters(word, offset, expected):
    assert offset_characters(word=word, offset=offset) == expected


def test_offset_characters_uses_full_alphabet():
    assert len(CHARS) == 26


def test_create_icl_example_structure():
    set_random_seed(0)
    adj_list = [[0, 1], [0, 2], [2, 3]]  # 3 chain steps over 3 inputs
    n_inputs = 3
    word_length = 4
    example = create_icl_example(
        word_length=word_length,
        num_words=n_inputs,
        adj_list=adj_list,
        char_offset=0,
    )
    assert set(example.keys()) == {"input_words", "intermediate_words", "answer_word", "adj_list"}
    assert len(example["input_words"]) == n_inputs
    # chain_length - 1 intermediates, 1 answer
    assert len(example["intermediate_words"]) == len(adj_list) - 1
    assert isinstance(example["answer_word"], str)
    assert example["adj_list"] == adj_list
    for word in example["input_words"]:
        assert len(word) == word_length


def test_create_icl_example_chain_words_are_deterministic_given_inputs():
    """With offset=0, the chain words must be a deterministic function of the
    input words and adjacency list."""
    set_random_seed(0)
    adj_list = [[0, 1]]
    example = create_icl_example(
        word_length=4,
        num_words=2,
        adj_list=adj_list,
        char_offset=0,
    )
    # answer word = second halves of input_words[0] + input_words[1]
    expected = split_and_concat_words(example["input_words"])
    assert example["answer_word"] == expected


def test_create_icl_example_applies_char_offset():
    set_random_seed(0)
    adj_list = [[0, 1]]
    no_offset = create_icl_example(word_length=4, num_words=2, adj_list=adj_list, char_offset=0)
    set_random_seed(0)
    with_offset = create_icl_example(word_length=4, num_words=2, adj_list=adj_list, char_offset=3)
    assert with_offset["input_words"] == no_offset["input_words"]
    assert with_offset["answer_word"] == offset_characters(no_offset["answer_word"], offset=3)


def test_get_icl_examples_count_and_consistency():
    set_random_seed(0)
    n_examples = 5
    n_inputs = 3
    n_parents = 2
    chain_length = 3
    examples = get_icl_examples(
        word_length=4,
        n_inputs=n_inputs,
        n_parents=n_parents,
        chain_length=chain_length,
        num_examples=n_examples,
        char_offset=1,
    )
    assert len(examples) == n_examples
    shared_adj_list = examples[0]["adj_list"]
    # The DAG is sampled once and shared across all examples in a task.
    for ex in examples:
        assert ex["adj_list"] == shared_adj_list
        assert len(ex["input_words"]) == n_inputs
        assert len(ex["intermediate_words"]) == chain_length - 1


def test_prepare_dataset_writes_to_output_dir(tmp_path):
    set_random_seed(0)
    out = tmp_path / "out"
    out.mkdir()
    n_tasks, n_eval_tasks = 2, 2
    ds = prepare_dataset(
        num_examples=2,
        word_length=3,
        n_inputs=2,
        n_parents=2,
        chain_length=2,
        char_offset=1,
        n_tasks=n_tasks,
        n_eval_tasks=n_eval_tasks,
        output_dir=str(out),
    )
    assert "train" in ds and "test" in ds
    assert len(ds["train"]) == n_tasks
    assert len(ds["test"]) == n_eval_tasks
    # one subdirectory should have been created under the output dir
    saved = os.listdir(out)
    assert len(saved) == 1
    assert saved[0].startswith("cot_icl_lab_symbolic_")
