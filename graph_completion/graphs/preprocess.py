"""
Module containing methods required for graph preprocessing.
"""
from queue import Queue
from typing import Callable, Dict, Generator, Iterable, List, Set, Tuple, Union

import attr
import igraph
import numpy as np
import pandas as pd
import torch as pt
from attr.validators import and_, deep_iterable, instance_of
from torch.nn.functional import one_hot
from tqdm import tqdm

from graph_completion.graphs.queries import check_negative, get_node_cut_cache, \
    Query, query_edge_r_to_int, QueryInstance
from graph_completion.graphs.random_walks import do_walk, obtain_context_indices
from graph_completion.utils import AbstractConf

AdjacencyIndex = Dict[int, Dict[int, Iterable[int]]]
Sample = Tuple[igraph.Graph, Iterable[int]]
SamplePair = Tuple[Query, Sample, Sample]


def get_efficient_indexes(node_data: pd.DataFrame, edge_data: pd.DataFrame,
                          community_membership: np.ndarray) -> Iterable[AdjacencyIndex]:
    nodes_of_community = {}
    for c, node_type, n in node_data.assign(c=community_membership)[["c", "type", "n"]].values:
        nodes_of_community.setdefault(c, {})
        nodes_of_community[c].setdefault(node_type, [])
        nodes_of_community[c][node_type].append(n)
    adjacency_source_to_target = {}
    adjacency_target_to_source = {}
    for s, r, t in edge_data[["s", "r", "t"]].values:
        adjacency_source_to_target.setdefault(s, {})
        adjacency_source_to_target[s].setdefault(r, [])
        adjacency_source_to_target[s][r].append(t)
        adjacency_target_to_source.setdefault(t, {})
        adjacency_target_to_source[t].setdefault(r, [])
        adjacency_target_to_source[t][r].append(s)
    edge_data_full = edge_data.assign(c_s=community_membership[edge_data.s],
                                      c_t=community_membership[edge_data.t])
    community_edge_data = edge_data_full[["c_s", "r", "c_t"]].drop_duplicates()
    del edge_data_full
    adjacency_source_to_target_c = {}
    adjacency_target_to_source_c = {}
    for c_s, r, c_t in community_edge_data[["c_s", "r", "c_t"]].values:
        adjacency_source_to_target_c.setdefault(c_s, {})
        adjacency_source_to_target_c[c_s].setdefault(r, [])
        adjacency_source_to_target_c[c_s][r].append(c_t)
        adjacency_target_to_source_c.setdefault(c_t, {})
        adjacency_target_to_source_c[c_t].setdefault(r, [])
        adjacency_target_to_source_c[c_t][r].append(c_s)
    return [nodes_of_community, adjacency_source_to_target, adjacency_target_to_source,
            adjacency_source_to_target_c, adjacency_target_to_source_c]


def get_neighbours_entity(adjacency: Dict[Tuple[int, int], Set[int]],
                          entity: int, relation: int, num_neighbours: int) -> np.ndarray:
    has_neighbours = (entity, relation) in adjacency
    neighbourhood = np.array(sorted(adjacency[entity, relation]) if has_neighbours else [entity, ])
    return np.random.choice(neighbourhood, size=num_neighbours, replace=len(neighbourhood) < num_neighbours)


def get_neighbours_query(entity_neighbours: Callable[[int, int], np.ndarray],
                         qi_tree: igraph.Graph, query: Query) -> np.ndarray:
    if "n" in qi_tree.es.attributes():
        del qi_tree.es["n"]
    query_node_stack = []
    for query_anchor in query.query_anchors:
        query_node_stack.append(query_anchor)
    while len(query_node_stack) > 0:
        query_node = query_node_stack.pop()
        query_parent_edges = qi_tree.es[qi_tree.incident(query_node, mode="in")]
        if len(query_parent_edges) == 0:
            continue
        query_parent_edge = query_parent_edges[0]
        query_parent_r = query_parent_edge["r"]
        if "p" in query_parent_r:
            query_parent = query_parent_edge.source
            relation = query_edge_r_to_int(query_parent_r)
            neighbours_node = entity_neighbours(qi_tree.vs[query_node]["e"], relation)
            neighbours_parent = entity_neighbours(qi_tree.vs[query_parent]["e"], relation)
            query_parent_edge["n"] = np.row_stack((neighbours_node, neighbours_parent))
            query_node_stack.append(query_parent)
        else:
            query_parent = query_parent_edge.source
            neighbours_node = qi_tree.es[qi_tree.incident(query_node, mode="out")][0]["n"][1]
            query_parent_edge["n"] = np.row_stack((neighbours_node, neighbours_node))
            query_node_stack.append(query_parent)
    return np.stack(qi_tree.es["n"])


class QueryData:
    def __init__(self, query: Query,
                 e: Union[pt.Tensor, List[pt.Tensor]] = None, x: Union[pt.Tensor, List[pt.Tensor]] = None,
                 c: Union[pt.Tensor, List[pt.Tensor]] = None, edge_attr: Union[pt.Tensor, List[pt.Tensor]] = None,
                 n: Union[pt.Tensor, List[pt.Tensor]] = None, e_c: Union[pt.Tensor, List[pt.Tensor]] = None,
                 edge_attr_c: Union[pt.Tensor, List[pt.Tensor]] = None, n_c: Union[pt.Tensor, List[pt.Tensor]] = None,
                 y: pt.Tensor = None, sample: pt.Tensor = None):
        self.query = query
        self.query_size = query.query_tree.vcount()

        query_tree_batch = query.query_tree.copy()
        for node_attr_name, node_attr in [("e", e), ("x", x), ("c", c), ("e_c", e_c)]:
            if node_attr is not None:
                query_tree_batch.vs[node_attr_name] = list(node_attr) if type(node_attr) is not list else node_attr
        for rel_attr_name, rel_attr in [("edge_attr", edge_attr), ("n", n), ("edge_attr_c", edge_attr_c), ("n_c", n_c)]:
            if rel_attr is not None:
                query_tree_batch.es[rel_attr_name] = list(rel_attr) if type(rel_attr) is not list else rel_attr
        for graph_attr_name, graph_attr in [("y", y), ("sample", sample)]:
            if graph_attr is not None:
                query_tree_batch[graph_attr_name] = graph_attr
        self.query_tree_batch = query_tree_batch

    def sample_split(self, mini_batch_size: int) -> Generator["QueryData", None, None]:
        sample = self.query_tree_batch["sample"]
        num_samples = sample.max().item() + 1
        for s in range(0, num_samples, mini_batch_size):
            batch_index = (s <= sample) & (sample < s + mini_batch_size)
            yield QueryData(self.query,
                            e=[e[batch_index] for e in self.query_tree_batch.vs["e"]],
                            x=[x[batch_index] for x in self.query_tree_batch.vs["x"]],
                            c=[c[batch_index] for c in self.query_tree_batch.vs["c"]],
                            edge_attr=[r[batch_index] for r in self.query_tree_batch.es["edge_attr"]],
                            n=[n[:, batch_index] for n in self.query_tree_batch.es["n"]],
                            e_c=[e_c[batch_index] for e_c in self.query_tree_batch.vs["e_c"]],
                            edge_attr_c=[r_c[batch_index] for r_c in self.query_tree_batch.es["edge_attr_c"]],
                            n_c=[n_c[:, batch_index] for n_c in self.query_tree_batch.es["n_c"]],
                            y=self.query_tree_batch["y"][batch_index],
                            sample=sample[batch_index] - s)

    def community_split(self) -> Generator[Tuple[pt.Tensor, "QueryData"], None, None]:
        c = pt.stack(self.query_tree_batch.vs["c"])
        device = c.device
        batch_keys, batch_splits_sizes = pt.unique_consecutive(c, dim=1, return_counts=True)
        if len(batch_splits_sizes) == 1:
            self.query_tree_batch.vs["x"] = self.query_tree_batch.vs["x_emb"]
            self.query_tree_batch.vs["e_c"] = self.query_tree_batch.vs["c_emb"]
            self.query_tree_batch.es["edge_attr_c"] = self.query_tree_batch.es["edge_attr_emb"]
            yield batch_keys[:, 0], self
        else:
            curr_query = pt.tensor(0, device=device)
            for c_tuple, batch_split_size in zip(batch_keys.T, batch_splits_sizes):
                yield c_tuple, QueryData(self.query,
                                         e=[e[curr_query:curr_query + batch_split_size]
                                            for e in self.query_tree_batch.vs["e"]],
                                         x=[x[curr_query:curr_query + batch_split_size]
                                            for x in self.query_tree_batch.vs["x_emb"]],
                                         edge_attr=[r[curr_query:curr_query + batch_split_size]
                                                    for r in self.query_tree_batch.es["edge_attr"]],
                                         n=[n[:, curr_query:curr_query + batch_split_size]
                                            for n in self.query_tree_batch.es["n"]],
                                         e_c=[e_c[curr_query:curr_query + batch_split_size]
                                              for e_c in self.query_tree_batch.vs["c_emb"]],
                                         edge_attr_c=[r_c[curr_query:curr_query + batch_split_size]
                                                      for r_c in self.query_tree_batch.es["edge_attr_emb"]])
                curr_query += batch_split_size

    def batch_split(self, mini_batch_size: int) -> Generator["QueryData", None, None]:
        num_samples = len(self)
        for b in range(0, num_samples, mini_batch_size):
            yield QueryData(self.query,
                            e=[e[b:b + mini_batch_size] for e in self.query_tree_batch.vs["e"]]
                            if "e" in self.query_tree_batch.vs.attributes() else None,
                            x=[x[b:b + mini_batch_size] for x in self.query_tree_batch.vs["x"]]
                            if "x" in self.query_tree_batch.vs.attributes() else None,
                            c=[c[b:b + mini_batch_size] for c in self.query_tree_batch.vs["c"]]
                            if "c" in self.query_tree_batch.vs.attributes() else None,
                            edge_attr=[r[b:b + mini_batch_size] for r in self.query_tree_batch.es["edge_attr"]]
                            if "edge_attr" in self.query_tree_batch.es.attributes() else None,
                            n=[n[:, b:b + mini_batch_size] for n in self.query_tree_batch.es["n"]]
                            if "n" in self.query_tree_batch.es.attributes() else None,
                            e_c=[e_c[b:b + mini_batch_size] for e_c in self.query_tree_batch.vs["e_c"]]
                            if "e_c" in self.query_tree_batch.vs.attributes() else None,
                            edge_attr_c=[r_c[b:b + mini_batch_size] for r_c in self.query_tree_batch.es["edge_attr_c"]]
                            if "edge_attr_c" in self.query_tree_batch.es.attributes() else None,
                            n_c=[n_c[:, b:b + mini_batch_size] for n_c in self.query_tree_batch.es["n_c"]]
                            if "n_c" in self.query_tree_batch.es.attributes() else None,
                            y=self.query_tree_batch["y"][b:b + mini_batch_size]
                            if "y" in self.query_tree_batch.attributes() else None,
                            sample=self.query_tree_batch["sample"][b:b + mini_batch_size]
                            if "sample" in self.query_tree_batch.attributes() else None)

    def get_only_positive(self) -> Generator["QueryData", None, None]:
        y = self.query_tree_batch["y"]
        positive_samples = QueryData(self.query,
                                     e=[e[y] for e in self.query_tree_batch.vs["e"]],
                                     x=[x[y] for x in self.query_tree_batch.vs["x"]],
                                     c=[c[y] for c in self.query_tree_batch.vs["c"]],
                                     edge_attr=[r[y] for r in self.query_tree_batch.es["edge_attr"]],
                                     n=[n[:, y] for n in self.query_tree_batch.es["n"]],
                                     e_c=[e_c[y] for e_c in self.query_tree_batch.vs["e_c"]],
                                     edge_attr_c=[r_c[y] for r_c in self.query_tree_batch.es["edge_attr_c"]],
                                     n_c=[n_c[:, y] for n_c in self.query_tree_batch.es["n_c"]],
                                     sample=self.query_tree_batch["sample"][y])
        return positive_samples.batch_split(1)

    def __len__(self) -> int:
        return len(self.query_tree_batch.vs["e"][0]) if "e" in self.query_tree_batch.vs.attributes() else (
            len(self.query_tree_batch.vs["e_c"][0]) if "e_c" in self.query_tree_batch.vs.attributes() else 0
        )

    def __repr__(self):
        result = ""
        for node_attr in ["e", "x", "c", "e_c"]:
            if node_attr not in self.query_tree_batch.vs.attributes():
                continue
            result += f"{node_attr}={[self.query_size, ] + list(self.query_tree_batch.vs[node_attr][0].size())}, "
        for rel_attr in ["edge_attr", "n", "edge_attr_c", "n_c"]:
            if rel_attr not in self.query_tree_batch.es.attributes():
                continue
            result += f"{rel_attr}={[self.query_size - 1, ] + list(self.query_tree_batch.es[rel_attr][0].size())}, "
        for graph_attr in ["y", "sample"]:
            if graph_attr not in self.query_tree_batch.attributes():
                continue
            result += f"{graph_attr}={list(self.query_tree_batch[graph_attr].size())}, "
        result = f"QueryData({result.rstrip(', ')})"
        return result


class QueryEmbeddingData:
    def __init__(self, c_query_emb: pt.Tensor, c_answer_emb: pt.Tensor,
                 query_emb: pt.Tensor, answer_emb: pt.Tensor,
                 y: pt.Tensor, sample: pt.Tensor):
        self.c_query_emb = c_query_emb
        self.c_answer_emb = c_answer_emb
        self.query_emb = query_emb
        self.answer_emb = answer_emb
        self.y = y
        self.sample = sample

    def __repr__(self):
        return f"QueryEmbeddingData(" \
               f"c_query_emb={list(self.c_query_emb.size())}, c_answer_emb={list(self.c_answer_emb.size())}, " \
               f"query_emb={list(self.query_emb.size())}, answer_emb={list(self.answer_emb.size())}, " \
               f"y={list(self.y.size())}, sample={list(self.sample.size())})"

    def __len__(self):
        return self.query_emb.size(0)

    def sample_split_embeddings(self) -> List[Tuple[pt.Tensor, pt.Tensor]]:
        batch_size = self.sample.max().item() + 1
        sample_sort_index = pt.argsort(self.sample)
        y = self.y[sample_sort_index]
        sample_emb = []
        for emb in [self.c_query_emb, self.c_answer_emb, self.query_emb, self.answer_emb]:
            emb = emb[..., sample_sort_index, :]
            pos_emb = emb[..., y, :]
            neg_emb = pt.stack(pt.chunk(emb[..., ~y, :], batch_size, dim=-2), dim=-3)
            sample_emb.append((pos_emb, neg_emb))
        return sample_emb


def samples_to_tensors(samples: List[SamplePair], node_types: np.ndarray,
                       communities: np.ndarray, num_communities: int,
                       com_neighbours: Callable[[int, int], np.ndarray],
                       node_neighbours: Callable[[int, int], np.ndarray],
                       num_node_types: int, num_relations: int, device: str) -> Dict[str, QueryData]:
    queries, sample_counters = {}, {}
    e, x, y, edge_attr, n, c, e_c, edge_attr_c, n_c, sample = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for query, (qi_tree, neg_answers), (qi_tree_c, neg_answers_c) in tqdm(samples,
                                                                          "Preprocessing samples", leave=False):
        queries.setdefault(query.structure, query)
        query_edge_frame = qi_tree.get_edge_dataframe()
        query_edge_s_t = query_edge_frame[["source", "target"]].values
        query_edge_r = query_edge_frame.r.map(query_edge_r_to_int).values
        query_edge_r[query_edge_r == -1] = num_relations
        query_edge_frame_c = qi_tree_c.get_edge_dataframe()
        query_edge_r_c = query_edge_frame_c.r.map(query_edge_r_to_int).values
        query_edge_r_c[query_edge_r_c == -1] = num_relations

        entities = qi_tree.vs["e"]
        e.setdefault(query.structure, [])
        e[query.structure].append(entities)
        edges = np.array(entities)[query_edge_s_t]
        x.setdefault(query.structure, [])
        x[query.structure].append(node_types[entities])
        y.setdefault(query.structure, [])
        y[query.structure].append(True)
        edge_attr.setdefault(query.structure, [])
        edge_attr[query.structure].append(query_edge_r)
        n_i = get_neighbours_query(node_neighbours, qi_tree, query)
        s, t, n_i_s, n_i_t = edges[:, 0], edges[:, 1], n_i[:, 0], n_i[:, 1]
        s_to_t_n_check = communities[n_i_s] != communities[t].reshape((-1, 1))
        t_to_s_n_check = communities[n_i_t] != communities[s].reshape((-1, 1))
        n_i[:, 0][s_to_t_n_check] = np.tile(s.reshape((-1, 1)), (1, n_i_s.shape[1]))[s_to_t_n_check]
        n_i[:, 1][t_to_s_n_check] = np.tile(t.reshape((-1, 1)), (1, n_i_t.shape[1]))[t_to_s_n_check]
        n.setdefault(query.structure, [])
        n[query.structure].append(n_i)
        c.setdefault(query.structure, [])
        c[query.structure].append(communities[entities])
        entities_c = qi_tree_c.vs["e"]
        e_c.setdefault(query.structure, [])
        e_c[query.structure].append(entities_c)
        edge_attr_c.setdefault(query.structure, [])
        edge_attr_c[query.structure].append(query_edge_r_c)
        n_c.setdefault(query.structure, [])
        n_c[query.structure].append(get_neighbours_query(com_neighbours, qi_tree_c, query))
        sample_counters.setdefault(query.structure, 0)
        sample.setdefault(query.structure, [])
        sample[query.structure].append(sample_counters[query.structure])
        for neg_answer, neg_answer_c in zip(neg_answers, neg_answers_c):
            qi_tree_neg = qi_tree.copy()
            qi_tree_neg.vs[query.query_answer]["e"] = neg_answer
            qi_tree_c_neg = qi_tree_c.copy()
            qi_tree_c_neg.vs[query.query_answer]["e"] = neg_answer_c

            entities = qi_tree_neg.vs["e"]
            e[query.structure].append(entities)
            edges = np.array(entities)[query_edge_s_t]
            x[query.structure].append(node_types[entities])
            y[query.structure].append(False)
            edge_attr[query.structure].append(query_edge_r)
            n_i = get_neighbours_query(node_neighbours, qi_tree_neg, query)
            s, t, n_i_s, n_i_t = edges[:, 0], edges[:, 1], n_i[:, 0], n_i[:, 1]
            s_to_t_n_check = communities[n_i_s] != communities[t].reshape((-1, 1))
            t_to_s_n_check = communities[n_i_t] != communities[s].reshape((-1, 1))
            n_i[:, 0][s_to_t_n_check] = np.tile(s.reshape((-1, 1)), (1, n_i_s.shape[1]))[s_to_t_n_check]
            n_i[:, 1][t_to_s_n_check] = np.tile(t.reshape((-1, 1)), (1, n_i_t.shape[1]))[t_to_s_n_check]
            n[query.structure].append(n_i)
            c[query.structure].append(communities[entities])
            entities_c = qi_tree_c_neg.vs["e"]
            e_c[query.structure].append(entities_c)
            edge_attr_c[query.structure].append(query_edge_r_c)
            n_c[query.structure].append(get_neighbours_query(com_neighbours, qi_tree_c_neg, query))
            sample[query.structure].append(sample_counters[query.structure])
        sample_counters[query.structure] += 1

    query_tree_batches = {}
    for query_structure in queries:
        c_q = pt.tensor(np.array(c[query_structure]), dtype=pt.long, device=device).T
        batch_sort_index = pt.argsort(
            pt.sum(c_q * (num_communities ** pt.arange(c_q.size(0), device=device)).view(-1, 1), dim=0)
        )
        e_q = pt.tensor(e[query_structure], dtype=pt.long, device=device)[batch_sort_index].T
        x_q = one_hot(
            pt.tensor(np.array(x[query_structure]), dtype=pt.long, device=device)[batch_sort_index].T,
            num_node_types).float()
        y_q = pt.tensor(y[query_structure], device=device)[batch_sort_index]
        edge_attr_q = one_hot(
            pt.tensor(np.array(edge_attr[query_structure]), dtype=pt.long, device=device)[batch_sort_index].T,
            num_relations + 1).float()
        n_q = pt.tensor(
            np.array(n[query_structure]), dtype=pt.long, device=device
        )[batch_sort_index].permute(1, 2, 0, 3)
        c_q = c_q[:, batch_sort_index]
        e_c_q = pt.tensor(e_c[query_structure], dtype=pt.long, device=device)[batch_sort_index].T
        edge_attr_c_q = one_hot(
            pt.tensor(np.array(edge_attr_c[query_structure]), dtype=pt.long, device=device)[batch_sort_index].T,
            num_relations + 1).float()
        n_c_q = pt.tensor(
            np.array(n_c[query_structure]), dtype=pt.long, device=device
        )[batch_sort_index].permute(1, 2, 0, 3)
        sample_q = pt.tensor(sample[query_structure], dtype=pt.long, device=device)[batch_sort_index]

        query_tree_batches[query_structure] = QueryData(queries[query_structure], e_q, x_q, c_q, edge_attr_q, n_q,
                                                        e_c_q, edge_attr_c_q, n_c_q, y_q, sample_q)

    return query_tree_batches


class Sampler:
    def __init__(self, query_structure: List[str], num_negative_samples: int, num_neighbours: int,
                 random_walk_length: int, context_radius: int,
                 pagerank_importances: bool, walks_relation_specific: bool):
        self.query_structure = query_structure
        self.num_negative_samples = num_negative_samples
        self.num_neighbours = num_neighbours
        self.random_walk_length = random_walk_length
        self.context_radius = context_radius
        self.pagerank_importances = pagerank_importances
        self.walks_relation_specific = walks_relation_specific

        self.num_query_structures: int = len(self.query_structure)
        self.queries: List[Query] = [Query(structure) for structure in self.query_structure]
        for query in self.queries:
            query.build_query_tree()
            query.get_node_cut()
        self.query_index: int = 0
        self.answer_indices: np.ndarray = np.zeros(self.num_query_structures, dtype=int)
        self.answer_indices_c: np.ndarray = np.zeros(self.num_query_structures, dtype=int)
        self.qi_generators: List[Generator[QueryInstance, None, None]] = \
            [None for _ in range(self.num_query_structures)]
        self.qi_c_generators: List[Generator[QueryInstance, None, None]] = \
            [None for _ in range(self.num_query_structures)]
        self.sample_queues: List["Queue[Sample]"] = [Queue() for _ in range(self.num_query_structures)]
        self.sample_queues_c: List["Queue[Sample]"] = [Queue() for _ in range(self.num_query_structures)]
        self.invalid_neg_targets: Dict[Tuple[int, int], Set[int]] = None

    def set_neg_filtering_index(self, val_edges: pd.DataFrame, test_edges: pd.DataFrame):
        self.invalid_neg_targets = {}
        for s, r, t in pd.concat((val_edges, test_edges), sort=True)[["s", "r", "t"]].values:
            self.invalid_neg_targets.setdefault((s, r), set())
            self.invalid_neg_targets[s, r].update([t, ])

    def get_neg_samples_random_walks(self, qi_mapped: igraph.Graph, query: Query, walk: List[int], index: int,
                                     n_of_c: pd.Series, community_membership: np.ndarray, num_communities: int,
                                     for_nodes: bool) -> Iterable[int]:
        anchor, answer = qi_mapped.vs[query.query_anchors[0]]["e"], qi_mapped.vs[query.query_answer]["e"]
        r = query_edge_r_to_int(qi_mapped.es[query.query_answer - 1]["r"])
        walk = np.array(walk, dtype=int)
        walk_context = walk[np.abs(index - np.arange(len(walk))) <= self.context_radius]
        negative_samples, negative_samples_c = [], []

        if for_nodes:
            c_anchor, c_answer = community_membership[anchor], community_membership[answer]
            candidate_pool = np.array(sum([nodes for node_type, nodes in n_of_c[c_answer].items()], []))
            invalid_neg_targets = set()
            if (anchor, r) in self.invalid_neg_targets:
                invalid_neg_targets.update(self.invalid_neg_targets[anchor, r])
            candidate_pool = candidate_pool[~np.isin(candidate_pool, invalid_neg_targets)]
        else:
            candidate_pool = np.arange(num_communities)
        candidate_pool = candidate_pool[~np.isin(candidate_pool, walk_context)]
        if len(candidate_pool) > 0:
            negative_samples = np.random.choice(candidate_pool, size=self.num_negative_samples)
        return negative_samples

    def get_neg_samples_smore(self, qi_mapped: igraph.Graph, query: Query, node_cut_cache: Dict[int, List[int]],
                              n_of_c: AdjacencyIndex, adj_t_to_s: AdjacencyIndex,
                              community_membership: np.ndarray, num_communities: int,
                              for_nodes: bool) -> Iterable[int]:
        anchors, answer = qi_mapped.vs[query.query_anchors]["e"], qi_mapped.vs[query.query_answer]["e"]
        r = query_edge_r_to_int(qi_mapped.es[query.query_answer - 1]["r"])
        negative_samples = []

        if for_nodes:
            c_answer = community_membership[answer]
            invalid_neg_targets = set()
            for anchor in anchors:
                if (anchor, r) in self.invalid_neg_targets:
                    invalid_neg_targets.update(self.invalid_neg_targets[anchor, r])
            candidate_pool = np.array(sum([nodes for node_type, nodes in n_of_c[c_answer].items()], []))
            candidate_pool = candidate_pool[~np.isin(candidate_pool, invalid_neg_targets)]
        else:
            candidate_pool = np.arange(num_communities)
        if len(candidate_pool) > 0:
            candidate_pool = np.random.permutation(candidate_pool)
            for candidate in candidate_pool:
                if check_negative(qi_mapped, query, node_cut_cache, adj_t_to_s, candidate):
                    negative_samples.append(candidate)
                    if len(negative_samples) == self.num_negative_samples:
                        break
            if 0 < len(negative_samples) < self.num_negative_samples:
                negative_samples = np.random.choice(negative_samples, size=self.num_negative_samples)
        return negative_samples

    def get_query_samples_random_walks(self, num_samples: int, graph_indexes: Iterable[AdjacencyIndex],
                                       num_nodes: int, num_relations: int,
                                       community_membership: np.ndarray, num_communities: int) -> Iterable[SamplePair]:
        n_of_c, adj_s_to_t, adj_t_to_s, adj_s_to_t_c, adj_t_to_s_c = graph_indexes
        samples = []

        query = self.queries[self.query_index]
        sample_queue, sample_queue_c = self.sample_queues[self.query_index], self.sample_queues_c[self.query_index]
        qi_generator, qi_c_generator = self.qi_generators[self.query_index], self.qi_c_generators[self.query_index]
        progress_bar = tqdm(range(num_samples), "Sampling query batch", leave=False)
        while len(samples) < num_samples:
            while sample_queue.empty():
                qi = None
                while qi is None:
                    answer = self.answer_indices[self.query_index]
                    if qi_generator is None:
                        qi_generator = query.instantiate(adj_t_to_s, num_nodes, num_relations, answer)
                    try:
                        qi = next(qi_generator)
                        qi_tree = qi.map_to_tree(query.query_tree)
                        r = query_edge_r_to_int(qi_tree.es[0]["r"])
                        walk_length, walk_nodes = do_walk(answer, r, self.random_walk_length, adj_t_to_s)
                        context_indices = [] if walk_length < 2 else obtain_context_indices(walk_length,
                                                                                            self.context_radius)
                        for node_index, context_node_index in context_indices:
                            qi_tree_walk = qi_tree.copy()
                            qi_tree_walk.vs[query.query_anchors[0]]["e"] = walk_nodes[node_index]
                            qi_tree_walk.vs[query.query_answer]["e"] = walk_nodes[context_node_index]
                            negative_samples = self.get_neg_samples_random_walks(qi_tree_walk, query,
                                                                                 walk_nodes, node_index, n_of_c,
                                                                                 community_membership, num_communities,
                                                                                 True)
                            if len(negative_samples) > 0:
                                sample_queue.put((qi_tree_walk, negative_samples))
                    except StopIteration:
                        qi_generator = None
                        self.answer_indices[self.query_index] = (answer + 1) % num_nodes

            while sample_queue_c.empty():
                qi_c = None
                while qi_c is None:
                    answer_c = self.answer_indices_c[self.query_index]
                    if qi_c_generator is None:
                        qi_c_generator = query.instantiate(adj_t_to_s_c, num_communities, num_relations, answer_c)
                    try:
                        qi_c = next(qi_c_generator)
                        qi_tree_c = qi_c.map_to_tree(query.query_tree)
                        r_c = query_edge_r_to_int(qi_tree_c.es[0]["r"])
                        walk_length, walk_nodes = do_walk(answer_c, r_c, self.random_walk_length, adj_t_to_s_c)
                        context_indices = [] if walk_length < 2 else obtain_context_indices(walk_length,
                                                                                            self.context_radius)
                        for node_index, context_node_index in context_indices:
                            qi_tree_walk_c = qi_tree_c.copy()
                            qi_tree_walk_c.vs[query.query_anchors[0]]["e"] = walk_nodes[node_index]
                            qi_tree_walk_c.vs[query.query_answer]["e"] = walk_nodes[context_node_index]
                            negative_samples_c = self.get_neg_samples_random_walks(qi_tree_walk_c, query,
                                                                                   walk_nodes, node_index, n_of_c,
                                                                                   community_membership,
                                                                                   num_communities, False)
                            if len(negative_samples_c) > 0:
                                sample_queue_c.put((qi_tree_walk_c, negative_samples_c))
                    except StopIteration:
                        qi_c_generator = None
                        self.answer_indices_c[self.query_index] = (answer_c + 1) % num_communities

            samples.append((query, sample_queue.get(), sample_queue_c.get()))
            progress_bar.update()
        progress_bar.close()

        self.qi_generators[self.query_index], self.qi_c_generators[self.query_index] = qi_generator, qi_c_generator
        self.query_index = (self.query_index + 1) % self.num_query_structures
        return samples

    def get_query_samples_smore(self, num_samples: int, graph_indexes: Iterable[AdjacencyIndex],
                                num_nodes: int, num_relations: int,
                                community_membership: np.ndarray, num_communities: int) -> Iterable[SamplePair]:
        n_of_c, adj_s_to_t, adj_t_to_s, adj_s_to_t_c, adj_t_to_s_c = graph_indexes
        samples = []

        query = self.queries[self.query_index]
        sample_queue, sample_queue_c = self.sample_queues[self.query_index], self.sample_queues_c[self.query_index]
        qi_generator, qi_c_generator = self.qi_generators[self.query_index], self.qi_c_generators[self.query_index]
        progress_bar = tqdm(range(num_samples), "Sampling query batch", leave=False)
        while len(samples) < num_samples:
            while sample_queue.empty():
                qi = None
                while qi is None:
                    answer = self.answer_indices[self.query_index]
                    if qi_generator is None:
                        qi_generator = query.instantiate(adj_t_to_s, num_nodes, num_relations, answer)
                    try:
                        qi = next(qi_generator)
                        qi_tree = qi.map_to_tree(query.query_tree)
                        node_cut_cache = get_node_cut_cache(qi_tree, query, adj_s_to_t)
                        negative_samples = self.get_neg_samples_smore(qi_tree, query, node_cut_cache,
                                                                      n_of_c, adj_t_to_s,
                                                                      community_membership, num_communities,
                                                                      True)
                        if len(negative_samples) > 0:
                            sample_queue.put((qi_tree, negative_samples))
                    except StopIteration:
                        qi_generator = None
                        self.answer_indices[self.query_index] = (answer + 1) % num_nodes

            while sample_queue_c.empty():
                qi_c = None
                while qi_c is None:
                    answer_c = self.answer_indices_c[self.query_index]
                    if qi_c_generator is None:
                        qi_c_generator = query.instantiate(adj_t_to_s_c, num_communities, num_relations, answer_c)
                    try:
                        qi_c = next(qi_c_generator)
                        qi_tree_c = qi_c.map_to_tree(query.query_tree)
                        node_cut_cache_c = get_node_cut_cache(qi_tree_c, query, adj_s_to_t_c)
                        negative_samples_c = self.get_neg_samples_smore(qi_tree_c, query, node_cut_cache_c,
                                                                        n_of_c, adj_t_to_s_c,
                                                                        community_membership, num_communities,
                                                                        False)
                        if len(negative_samples_c) > 0:
                            sample_queue_c.put((qi_tree_c, negative_samples_c))
                    except StopIteration:
                        qi_c_generator = None
                        self.answer_indices_c[self.query_index] = (answer_c + 1) % num_communities

            samples.append((query, sample_queue.get(), sample_queue_c.get()))
            progress_bar.update()
        progress_bar.close()

        self.qi_generators[self.query_index], self.qi_c_generators[self.query_index] = qi_generator, qi_c_generator
        self.query_index = (self.query_index + 1) % self.num_query_structures
        return samples


@attr.s
class SamplerHpars(AbstractConf):
    OPTIONS = {"sampler": Sampler}
    query_structure = attr.ib(default=["1p", ], validator=deep_iterable(
        instance_of(str), and_(instance_of(list), attr.validators.min_len(1))
    ))
    num_negative_samples = attr.ib(default=5, validator=instance_of(int))
    num_neighbours = attr.ib(default=10, validator=instance_of(int))
    random_walk_length = attr.ib(default=10, validator=instance_of(int))
    context_radius = attr.ib(default=2, validator=instance_of(int))
    pagerank_importances = attr.ib(default=True, validator=instance_of(bool))
    walks_relation_specific = attr.ib(default=True, validator=instance_of(bool))
    name = "sampler"
