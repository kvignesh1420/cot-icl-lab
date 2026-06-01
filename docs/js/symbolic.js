// Track B — natural-language symbolic construction.
//
// Faithful port of src/tokenized_cot_icl/core/data/symbolic/prepare.py and the
// chat framing in symbolic/symbolic_data_module.py + symbolic/common.py.
//
// The same DAG skeleton is realized with *real words*: each chain word is built
// by taking the second half of every parent word, concatenating them, and
// Caesar-shifting by `charOffset`.

import { RNG } from "./rng.js";

const CHARS = "abcdefghijklmnopqrstuvwxyz".split("");

// --- common.py constants (verbatim) ---
export const THINK_PREFIX = "<|im_start|>think\n";
export const ANSWER_PREFIX = "<|im_start|>final answer\n";
export const SYSTEM_PROMPT = `You are a helpful problem solver who follows instructions.

Given some examples of string transformations, you must figure out the underlying transformation rule
and apply it to the set of words in the last example.

Instructions for the response:
a. If you choose to think out loud, you must start your response with: ${THINK_PREFIX}.
b. The final answer must always be provided in the: ${ANSWER_PREFIX}\\boxed{} format.
c. You should not think or explain after providing the final answer.
`;

export function createWord(length, rng) {
  let w = "";
  for (let i = 0; i < length; i++) w += rng.choice(CHARS);
  return w;
}

// prepare.py: word[len(word)//2:] for each word, concatenated.
export function secondHalf(word) {
  return word.slice(Math.floor(word.length / 2));
}

export function splitAndConcatWords(words) {
  return words.map(secondHalf).join("");
}

// prepare.py: shift each char by offset, modulo alphabet length.
export function offsetCharacters(word, offset) {
  return word
    .split("")
    .map((c) => CHARS[(CHARS.indexOf(c) + offset) % CHARS.length])
    .join("");
}

// Build one ICL example over a given adjacency list, recording per-step detail
// for the step-through visualization.
export function createIclExample({ wordLength, numWords, adjList, charOffset }, rng) {
  const inputWords = [];
  for (let i = 0; i < numWords; i++) inputWords.push(createWord(wordLength, rng));

  const all = inputWords.slice();
  const chainWords = [];
  const steps = [];

  for (const parents of adjList) {
    const parentWords = parents.map((idx) => all[idx]);
    const halves = parentWords.map(secondHalf);
    const concat = halves.join("");
    const chainWord = offsetCharacters(concat, charOffset);
    chainWords.push(chainWord);
    all.push(chainWord);
    steps.push({ parents, parentWords, halves, concat, chainWord });
  }

  return {
    inputWords,
    intermediateWords: chainWords.slice(0, -1),
    answerWord: chainWords[chainWords.length - 1],
    chainWords,
    adjList,
    steps,
  };
}

// --- chat framing (symbolic_data_module.py) ---

export function prepareQuestion(inputWords) {
  return (
    "Given the input words: " +
    inputWords.join(", ") +
    ", what is the final answer after the string transformations?"
  );
}

export function prepareCotSolution(intermediateWords, answerWord) {
  let reasoning = THINK_PREFIX;
  for (let i = 0; i < intermediateWords.length; i++) {
    reasoning += `Step ${i + 1}: ${intermediateWords[i]}\n`;
  }
  reasoning += `${ANSWER_PREFIX}\\boxed{${answerWord}}`;
  return reasoning;
}

export function prepareDirectSolution(answerWord) {
  return `${ANSWER_PREFIX}\\boxed{${answerWord}}`;
}

// A conversation is an array of {role, content}. Rendered with Qwen2.5 / ChatML
// markers, matching tokenizer.apply_chat_template.
export function buildConversation(examples, cotFlags) {
  const conversation = [{ role: "system", content: SYSTEM_PROMPT }];
  examples.forEach((ex, i) => {
    conversation.push({ role: "user", content: prepareQuestion(ex.inputWords) });
    const assistant = cotFlags[i]
      ? prepareCotSolution(ex.intermediateWords, ex.answerWord)
      : prepareDirectSolution(ex.answerWord);
    conversation.push({ role: "assistant", content: assistant });
  });
  return conversation;
}

export function renderChatML(conversation) {
  return conversation
    .map((m) => `<|im_start|>${m.role}\n${m.content}<|im_end|>`)
    .join("\n");
}

export { CHARS };
