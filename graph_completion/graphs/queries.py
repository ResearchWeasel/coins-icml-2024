import itertools
from queue import Queue
from typing import Dict, Generator, List, Optional, Tuple, TYPE_CHECKING

import igraph
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from graph_completion.graphs.preprocess import AdjacencyIndex


def query_edge_r_to_int(query_edge_r: str) -> int:
    return int(query_edge_r[1:]) if "p" in query_edge_r else -1


class QueryInstance:
    def __init__(self, query_node: int, entity: int):
        self.query_node = query_node
        self.entity = entity
        self.relation: str = ""
        self.children: List["QueryInstance"] = []
        self.query_size = 1

    def add_child(self, relation: str, child: "QueryInstance"):
        self.relation = relation
        self.children.append(child)
        self.query_size += child.query_size

    def map_to_tree(self, query_tree: igraph.Graph) -> igraph.Graph:
        def map_to_tree_r(query_instance: "QueryInstance"):
            query_entities[query_instance.query_node] = query_instance.entity
            for child in query_instance.children:
                query_tree_mapped.es[child.query_node]["r"] = query_instance.relation
                map_to_tree_r(child)

        query_tree_mapped: igraph.Graph = query_tree.copy()
        query_entities = np.zeros(self.query_size, dtype=int)
        map_to_tree_r(self)
        query_tree_mapped.vs["e"] = query_entities
        return query_tree_mapped

    def __repr__(self, level: int = 0) -> str:
        level_pad = '\t' * level
        res = f"{level_pad}{self.query_node}:{self.entity}"
        res += f"--{self.relation}-->\n" if self.relation else "\n"
        for child in self.children:
            res += child.__repr__(level + 1)
        return res


class Query:
    def __init__(self, structure: str):
        self.structure = structure
        self.query_tree: igraph.Graph = None
        self.query_anchors: List[int] = None
        self.query_answer: int = None
        self.query_node_cut: List[int] = None

    def build_query_tree(self):
        query_edges = []
        n = len(self.structure)
        curr_node = 0
        for i in range(0, n, 2):
            num_relations, relation_type = int(self.structure[i]), self.structure[i + 1]
            branching = relation_type != "p"
            for j in range(num_relations):
                if not branching:
                    query_edges.append((curr_node + j + 1, curr_node + j, "p"))
                else:
                    query_edges.append((curr_node + 2 * j + 1, curr_node + 2 * j, "p"))
                    query_edges.append((curr_node + 2 * num_relations, curr_node + 2 * j + 1, relation_type))
            curr_node += 2 * num_relations if branching else num_relations
        query_edges = pd.DataFrame(query_edges, columns=["source", "target", "r"])
        self.query_tree = igraph.Graph.DataFrame(query_edges, directed=True)
        out_degrees, in_degrees = np.array(self.query_tree.outdegree()), np.array(self.query_tree.indegree())
        self.query_anchors = np.nonzero(out_degrees == 0)[0].tolist()
        self.query_answer = np.nonzero(in_degrees == 0)[0][0]

    def get_node_cut(self):
        def compute_s(v: int):
            w_edges = self.query_tree.es[self.query_tree.incident(v, mode="out")]
            if len(w_edges) == 0:
                return
            children_values = []
            for w_edge in w_edges:
                w, w_r = w_edge.target, w_edge["r"]
                compute_s(w)
                children_values.append(s_cost[w] + (w_r == "p"))
            s_cost[v] = max(children_values)

        def compute_o(v: int):
            children = self.query_tree.neighbors(v, mode="out")
            if len(children) == 0:
                o_cost[v] = u_cost[v]
                return
            children_values = []
            for w in children:
                compute_o(w)
                children_values.append(o_cost[w])
            o_cost[v] = min(max(children_values), max(u_cost[v], s_cost[v]))

        def compute_node_cut(v: int):
            children = self.query_tree.neighbors(v, mode="out")
            if len(children) == 0:
                return
            children_values = o_cost[children]
            if max(children_values) >= max(u_cost[v], s_cost[v]):
                node_cut.append(v)
                return
            else:
                for w in children:
                    compute_node_cut(w)

        query_size = self.query_tree.vcount()
        u_cost, s_cost, o_cost = np.zeros(query_size), np.zeros(query_size), np.zeros(query_size)
        node_cut = []

        # Compute u
        node_queue = Queue()
        node_queue.put(self.query_answer)
        while not node_queue.empty():
            node = node_queue.get()
            child_edges = self.query_tree.es[self.query_tree.incident(node, mode="out")]
            for child_edge in child_edges:
                child, r = child_edge.target, child_edge["r"]
                u_cost[child] = u_cost[node] + (r == "p")
                node_queue.put(child)

        # Compute s
        compute_s(self.query_answer)

        # Compute o
        compute_o(self.query_answer)

        # Compute node cut
        compute_node_cut(self.query_answer)
        self.query_node_cut = node_cut

    def instantiate(self, adj_t_to_s: "AdjacencyIndex", num_entities: int, num_relations: int, answer: int,
                    parent_r: Optional[Tuple[int, int]] = None) -> Generator[QueryInstance, None, None]:
        def instantiate_r(query_node: int, entity: int, parent_r: Optional[Tuple[int, int]] = None,
                          negate: bool = False) -> Generator[QueryInstance, None, None]:
            query_child_edges = self.query_tree.es[self.query_tree.incident(query_node, mode="out")]
            if len(query_child_edges) == 0:
                yield QueryInstance(query_node, entity)
                return
            query_children_r = query_child_edges[0]["r"]
            if query_children_r == "p":
                if negate:
                    while True:
                        next_relation_random = np.random.choice(num_relations, size=1)[0]
                        next_entity_random = np.random.choice(num_entities, size=1)[0]
                        if entity not in adj_t_to_s or next_relation_random not in adj_t_to_s[entity]:
                            break
                        if next_entity_random not in adj_t_to_s[entity][next_relation_random]:
                            break
                    entity_relations = {next_relation_random: [next_entity_random, ]}
                else:
                    if entity not in adj_t_to_s:
                        return
                    entity_relations = adj_t_to_s[entity]
                query_child = query_child_edges[0].target
                for child_relation, next_entity_candidates in entity_relations.items():
                    for next_entity_candidate in next_entity_candidates:
                        if (entity == answer and parent_r is not None
                                and (child_relation, next_entity_candidate) != parent_r):
                            continue
                        for child_query_instance in instantiate_r(query_child, next_entity_candidate, None):
                            query_instance = QueryInstance(query_node, entity)
                            query_instance.add_child(f"p{child_relation}", child_query_instance)
                            yield query_instance
            elif query_children_r == "i":
                intersection_bins = []
                for i, query_child_edge in enumerate(query_child_edges):
                    query_child = query_child_edge.target
                    intersection_bins.append(instantiate_r(query_child, entity, parent_r if i == 0 else None))
                qi_tuple_equivalence_classes = None
                for qi_tuple in itertools.product(*intersection_bins):
                    qi_tuple_next_entities = [qi.children[0].entity for qi in qi_tuple]
                    if len(qi_tuple_next_entities) != len(set(qi_tuple_next_entities)):
                        continue
                    if qi_tuple_equivalence_classes is None:
                        qi_tuple_query_sizes = np.array([qi.query_size for qi in qi_tuple])
                        qi_tuple_equivalence_classes = np.array(igraph.Graph.Adjacency(
                            qi_tuple_query_sizes.reshape((-1, 1)) == qi_tuple_query_sizes.reshape((1, -1)),
                            mode="undirected").clusters("weak").membership)
                        qi_tuple_num_equivalence_classes = qi_tuple_equivalence_classes.max() + 1
                        qi_tuple_equivalence_classes = [np.nonzero(qi_tuple_equivalence_classes == i)[0]
                                                        for i in range(qi_tuple_num_equivalence_classes)]
                    qi_tuple_next_entities_sorted = np.array(qi_tuple_next_entities)
                    for qi_tuple_equivalence_class in qi_tuple_equivalence_classes:
                        qi_tuple_next_entities_sorted[qi_tuple_equivalence_class] = \
                            np.sort(qi_tuple_next_entities_sorted[qi_tuple_equivalence_class])
                    if np.any(qi_tuple_next_entities_sorted != np.array(qi_tuple_next_entities)):
                        continue
                    query_instance = QueryInstance(query_node, entity)
                    for qi in qi_tuple:
                        query_instance.add_child("i", qi)
                    yield query_instance
            else:
                assert query_children_r in ("d", "b")
                negation_index = 0 if query_children_r == "b" else len(query_child_edges) - 1
                parent_r_index = len(query_child_edges) - 1 - negation_index
                difference_bins = []
                for i, query_child_edge in enumerate(query_child_edges):
                    query_child = query_child_edge.target
                    difference_bins.append(instantiate_r(query_child, entity,
                                                         parent_r if i == parent_r_index else None,
                                                         i == negation_index))
                qi_tuple_equivalence_classes = None
                for qi_tuple in itertools.product(*difference_bins):
                    qi_tuple_next_entities = [qi.children[0].entity for qi in qi_tuple]
                    if len(qi_tuple_next_entities) != len(set(qi_tuple_next_entities)):
                        continue
                    if qi_tuple_equivalence_classes is None:
                        qi_tuple_query_sizes = np.array([qi.query_size for qi in qi_tuple])
                        qi_tuple_equivalence = (qi_tuple_query_sizes.reshape((-1, 1))
                                                == qi_tuple_query_sizes.reshape((1, -1)))
                        qi_tuple_equivalence[negation_index] = qi_tuple_equivalence[:, negation_index] = False
                        qi_tuple_equivalence[negation_index, negation_index] = True
                        qi_tuple_equivalence_classes = np.array(igraph.Graph.Adjacency(
                            qi_tuple_equivalence, mode="undirected").clusters("weak").membership)
                        qi_tuple_num_equivalence_classes = qi_tuple_equivalence_classes.max() + 1
                        qi_tuple_equivalence_classes = [np.nonzero(qi_tuple_equivalence_classes == i)[0]
                                                        for i in range(qi_tuple_num_equivalence_classes)]
                    qi_tuple_next_entities_sorted = np.array(qi_tuple_next_entities)
                    for qi_tuple_equivalence_class in qi_tuple_equivalence_classes:
                        qi_tuple_next_entities_sorted[qi_tuple_equivalence_class] = \
                            np.sort(qi_tuple_next_entities_sorted[qi_tuple_equivalence_class])
                    if np.any(qi_tuple_next_entities_sorted != np.array(qi_tuple_next_entities)):
                        continue
                    query_instance = QueryInstance(query_node, entity)
                    for qi in qi_tuple:
                        query_instance.add_child(query_children_r, qi)
                    yield query_instance

        yield from instantiate_r(self.query_answer, answer, parent_r)

    def __eq__(self, other):
        if type(other) is not Query:
            return False
        else:
            return self.structure == other.structure

    def __hash__(self):
        return hash(self.structure)

    def __repr__(self):
        return self.structure


def get_node_cut_cache(query_instance_mapped: igraph.Graph, query: Query,
                       adj_s_to_t: "AdjacencyIndex") -> Dict[int, List[int]]:
    node_cut_cache = {}

    anchors = query_instance_mapped.vs[query.query_anchors]["e"]
    query_entity_queue = Queue()
    branching_semaphore = {}
    for query_anchor, anchor in zip(query.query_anchors, anchors):
        query_entity_queue.put((query_anchor, anchor))
    while not query_entity_queue.empty():
        query_node, entity = query_entity_queue.get()
        if query_node in query.query_node_cut:
            node_cut_cache.setdefault(query_node, [])
            node_cut_cache[query_node].append(entity)
        else:
            query_parent_edge = query_instance_mapped.es[
                query_instance_mapped.incident(query_node, mode="in")
            ][0]
            if "p" in query_parent_edge["r"]:
                query_parent = query_parent_edge.source
                entity_relation = query_edge_r_to_int(query_parent_edge["r"])
                if entity not in adj_s_to_t:
                    continue
                if entity_relation not in adj_s_to_t[entity]:
                    continue
                for next_entity_candidate in adj_s_to_t[entity][entity_relation]:
                    query_entity_queue.put((query_parent, next_entity_candidate))
            else:
                query_parent = query_parent_edge.source
                branching_children = query_instance_mapped.neighbors(query_parent, mode="out")
                branching_semaphore_key = (query_parent_edge["r"], query_parent, entity)
                branching_semaphore.setdefault(branching_semaphore_key, np.zeros(len(branching_children), dtype=bool))
                branch_index = branching_children.index(query_node)
                branching_semaphore[branching_semaphore_key][branch_index] = True
        if query_entity_queue.empty():
            for branching_semaphore_key, flags in branching_semaphore.items():
                branch_type, query_parent, entity = branching_semaphore_key
                if branch_type == "d":
                    flags[-1] = not flags[-1]
                elif branch_type == "b":
                    flags[0] = not flags[0]
                if np.all(branching_semaphore[branching_semaphore_key]):
                    query_entity_queue.put((query_parent, entity))
            branching_semaphore = {}
    for query_node in query.query_node_cut:
        node_cut_cache.setdefault(query_node, [])

    return node_cut_cache


def check_negative(query_instance_mapped: igraph.Graph, query: Query, node_cut_cache: Dict[int, List[int]],
                   adj_t_to_s: "AdjacencyIndex", candidate: int) -> bool:
    is_sample = True

    query_entity_stack = [(query.query_answer, candidate)]
    while len(query_entity_stack) > 0:
        query_node, entity = query_entity_stack.pop()
        if query_node in query.query_node_cut:
            if entity in node_cut_cache[query_node]:
                is_sample = False
                break
        else:
            query_child_edges = query_instance_mapped.es[query_instance_mapped.incident(query_node, mode="out")]
            query_children_r = query_child_edges[0]["r"]
            if "p" in query_children_r:
                query_child = query_child_edges[0].target
                entity_relation = query_edge_r_to_int(query_child_edges[0]["r"])
                if entity not in adj_t_to_s:
                    continue
                if entity_relation not in adj_t_to_s[entity]:
                    continue
                for next_entity_candidate in adj_t_to_s[entity][entity_relation]:
                    query_entity_stack.append((query_child, next_entity_candidate))
            else:
                for query_child_edge in query_child_edges:
                    query_child = query_child_edge.target
                    query_entity_stack.append((query_child, entity))

    return is_sample


def get_all_answers(query_instance_mapped: igraph.Graph, query: Query,
                    adj_s_to_t: "AdjacencyIndex") -> Generator[int, None, None]:
    anchors = query_instance_mapped.vs[query.query_anchors]["e"]
    query_entity_queue = Queue()
    branching_semaphore = {}
    for query_anchor, anchor in zip(query.query_anchors, anchors):
        query_entity_queue.put((query_anchor, anchor))
    while not query_entity_queue.empty():
        query_node, entity = query_entity_queue.get()
        query_parent_edges = query_instance_mapped.es[
            query_instance_mapped.incident(query_node, mode="in")
        ]
        if len(query_parent_edges) == 0:
            yield entity
            continue
        query_parent_edge = query_parent_edges[0]
        if "p" in query_parent_edge["r"]:
            query_parent = query_parent_edge.source
            entity_relation = query_edge_r_to_int(query_parent_edge["r"])
            if entity not in adj_s_to_t:
                continue
            if entity_relation not in adj_s_to_t[entity]:
                continue
            if query_parent == query.query_answer:
                for next_entity_candidate in adj_s_to_t[entity][entity_relation]:
                    query_entity_queue.put((query_parent, next_entity_candidate))
            else:
                query_entity_queue.put((query_parent, query_instance_mapped.vs[query_parent]["e"]))
        else:
            query_parent = query_parent_edge.source
            branching_children = query_instance_mapped.neighbors(query_parent, mode="out")
            branching_semaphore_key = (query_parent_edge["r"], query_parent, entity)
            branching_semaphore.setdefault(branching_semaphore_key, np.zeros(len(branching_children), dtype=bool))
            branch_index = branching_children.index(query_node)
            branching_semaphore[branching_semaphore_key][branch_index] = True
        if query_entity_queue.empty():
            for branching_semaphore_key, flags in branching_semaphore.items():
                branch_type, query_parent, entity = branching_semaphore_key
                if branch_type == "d":
                    flags[-1] = not flags[-1]
                elif branch_type == "b":
                    flags[0] = not flags[0]
                if np.all(branching_semaphore[branching_semaphore_key]):
                    query_entity_queue.put((query_parent, entity))
            branching_semaphore = {}
