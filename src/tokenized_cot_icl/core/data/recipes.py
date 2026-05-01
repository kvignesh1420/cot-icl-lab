"""Recipe for the cot_example_prob parameter in the hybrid special token strategy."""

import abc
import math


class BaseRecipe(abc.ABC):
    def __init__(self, initial_prob: float, final_prob: float) -> None:
        self.initial_prob = initial_prob
        self.final_prob = final_prob

    @abc.abstractmethod
    def get_value(self, prompt_index: int) -> float:
        """Utilize the recipe configuration and logic to return the mapped value for a prompt_index"""
        ...


class PowerLawRecipe(BaseRecipe):
    """Power law interpolation between `initial_prob` and `final_prob` based on `n_prompts` (Both inclusive).

    We want to return:
    - `initial_prob` for first prompt index and
    - `final_prob` for the last prompt index

    The values are given based on the function:
    >>> y = c1 * math.pow(x, alpha) + c2

    Where:
    >>> x = prompt_index / n_prompts

    """

    def __init__(
        self,
        initial_prob: float,
        final_prob: float,
        n_prompts: int,
        alpha: float,
        scale: float,
        **kwargs,
    ) -> None:
        super().__init__(initial_prob, final_prob)
        self.n_prompts = n_prompts
        self.alpha = alpha
        self.scale = scale

    def get_value(self, prompt_index: int) -> float:
        x = prompt_index / self.n_prompts
        return min(self.scale * math.pow(x, self.alpha) + self.initial_prob, self.final_prob)


RECIPE_REGISTRY = {
    "power_law": PowerLawRecipe,
}
