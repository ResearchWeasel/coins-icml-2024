"""
Module containing the implementation of the context extraction process.
"""
from typing import List, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from graph_completion.graphs.preprocess import AdjacencyIndex


def do_walk(start: int, walk_relation: int, max_walk_length: int,
            adj_t_to_s: "AdjacencyIndex") -> Tuple[int, List[int]]:
    curr_node = start
    walk_length = 1
    walk_nodes = [curr_node, ]
    while walk_length < max_walk_length:
        if curr_node in adj_t_to_s and walk_relation in adj_t_to_s[curr_node]:
            curr_out_neighborhood = adj_t_to_s[curr_node][walk_relation]
        else:
            curr_out_neighborhood = []

        num_neighbours = len(curr_out_neighborhood)
        if num_neighbours == 0:
            break
        curr_node = np.random.choice(curr_out_neighborhood, size=1)[0]
        walk_nodes.append(curr_node)
        walk_length += 1
    walk_nodes = list(reversed(walk_nodes))
    return walk_length, walk_nodes


def obtain_context_indices(walk_length: int, context_radius: int) -> List[Tuple[int, int]]:
    """
    Calculate the position index pairs for every possible node-context pair in a random walk

    :param walk_length: length of the random walk
    :param context_radius: context size in one direction

    :return: list of position index pairs
    """

    node_context_indices = [(i, max(i - context_radius, 0), min(i + context_radius, walk_length - 1) + 1)
                            for i in range(walk_length)]
    context_triplet_indices = [(node_index, context_index)
                               for node_index, context_start, context_end in node_context_indices
                               for context_index in range(context_start, context_end) if node_index != context_index]
    return context_triplet_indices
