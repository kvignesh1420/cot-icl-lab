from tokenized_cot_icl.core.data.recipes import RECIPE_REGISTRY, BaseRecipe

DATA = {
    "n_input_choices": (3,),
    "chain_length_choices": (3,),
    "examples": [
        {"input_tokens": [1, 2, 3], "chain_tokens": [4, 5, 6]},
        {"input_tokens": [7, 8, 9], "chain_tokens": [10, 11, 12]},
    ],
}


def test_recipes_list():
    assert set(RECIPE_REGISTRY.keys()) == {"power_law"}


def test_power_law_cot_example_prob_recipe():
    recipe_clz: BaseRecipe = RECIPE_REGISTRY["power_law"]
    initial_prob = 0.0
    final_prob = 1.0
    n_prompts = 10
    alpha = 2
    scale = 1.0
    recipe = recipe_clz(
        initial_prob=initial_prob,
        final_prob=final_prob,
        n_prompts=n_prompts,
        alpha=alpha,
        scale=scale,
    )
    current_probs = []
    for prompt_index in range(n_prompts):
        current_probs.append(recipe.get_value(prompt_index=prompt_index))
    assert current_probs[0] == 0.0
    assert current_probs[n_prompts - 1] < 1
    assert current_probs[n_prompts // 2] < (final_prob - initial_prob) / 2
