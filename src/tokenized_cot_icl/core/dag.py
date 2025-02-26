"""This module contains the DAG class to define the relationships between the tokens"""

import numpy as np


class DAG:
    """
    A Directed Acyclic Graph (DAG) to define the relationships between the tokens.
    Given `n_inputs`, `n_parents`, and a `chain_length` of tokens,
    1. the DAG will have n_inputs + chain_length nodes.
    2. The first `n_inputs` nodes are the input tokens.
    3. The next `chain_length` nodes are the output tokens.
    4. Each chain token is dependent on `n_parents` tokens from the input tokens
        and all previously generated chain tokens.

    We return the adjancency list representation of the DAG for every chain token.
    For example:
    ```py
    >> n_inputs = 4
    >> n_parents = 4
    >> chain_length = 3
    >> dag = DAG(n_inputs, n_parents, chain_length)
    >> dag.generate_adj_list()

    [[2, 3, 0, 1], [2, 4, 3, 0], [4, 3, 0, 1]]
    ```

    **IMPORTANT:** Notice that the entries here correspond to the indices of the input tokens and not
    the actual tokens themselves.
    """

    def __init__(self, n_inputs: int, n_parents: int, chain_length: int):
        self.n_inputs = n_inputs
        self.n_parents = n_parents
        self.chain_length = chain_length

    def generate_adj_list(self) -> list:
        """
        Generate the adjacency list representation of the DAG.
        """
        adj_list = []
        available_indices = list(range(self.n_inputs))
        for chain_idx in range(self.chain_length):
            parent_indices = np.random.choice(
                available_indices, self.n_parents, replace=False
            ).tolist()
            adj_list.append(parent_indices)
            available_indices.append(self.n_inputs + chain_idx)
        return adj_list


# Register the DAG class

DAG_REGISTRY = {
    "random": DAG,
}
