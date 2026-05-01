import pytest

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


def test_power_law_caps_at_final_prob():
    """When the curve overshoots, get_value should clamp to final_prob."""
    recipe = RECIPE_REGISTRY["power_law"](
        initial_prob=0.5,
        final_prob=0.9,
        n_prompts=10,
        alpha=1,
        scale=10.0,  # large scale forces an early overshoot
    )
    # At index 1, raw value would be 0.5 + 10 * (1/10) = 1.5; clamped to 0.9.
    assert recipe.get_value(prompt_index=1) == 0.9
    assert recipe.get_value(prompt_index=9) == 0.9


def test_power_law_respects_initial_prob():
    recipe = RECIPE_REGISTRY["power_law"](
        initial_prob=0.25,
        final_prob=1.0,
        n_prompts=10,
        alpha=2,
        scale=1.0,
    )
    # At prompt_index=0, x=0 so the curve term is 0 -> initial_prob.
    assert recipe.get_value(prompt_index=0) == 0.25


def test_power_law_alpha_one_is_linear():
    recipe = RECIPE_REGISTRY["power_law"](
        initial_prob=0.0,
        final_prob=1.0,
        n_prompts=10,
        alpha=1,
        scale=1.0,
    )
    # y = x; at index 5, x=0.5 -> 0.5
    assert recipe.get_value(prompt_index=5) == pytest.approx(0.5)


def test_power_law_monotonically_non_decreasing():
    recipe = RECIPE_REGISTRY["power_law"](
        initial_prob=0.0,
        final_prob=1.0,
        n_prompts=20,
        alpha=2,
        scale=1.0,
    )
    values = [recipe.get_value(prompt_index=i) for i in range(20)]
    assert all(b >= a for a, b in zip(values, values[1:]))
