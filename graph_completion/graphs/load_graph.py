"""
Module containing methods for creating graph objects, from an edge dataset or using the algorithm for simulation.
"""
from glob import glob
from os import makedirs
from os.path import isfile, sep as os_path_sep
from typing import Dict, Iterable, Optional, Set, Tuple

import attr
import numpy as np
import pandas as pd
from attr.validators import in_
from tqdm import tqdm

from graph_analysis.metrics import Connectivity, Graph, Heterogeneity, Importance, Subgraphing
from graph_completion.graphs.preprocess import AdjacencyIndex, get_efficient_indexes, get_neighbours_entity, \
    QueryData, Sampler, SamplerHpars, samples_to_tensors
from graph_completion.graphs.queries import get_node_cut_cache
from graph_completion.utils import AbstractConf
from graph_data.load_data import Dataset, FreeBase, NELL, OGBBioKG, Transport, WordNet
from graph_data.serialization import load_object, save_object


class Loader:
    datasets = {"ogbl-biokg": OGBBioKG(), "freebase": FreeBase(), "wordnet": WordNet(), "nell": NELL()}
    transport_cities = [path.split(os_path_sep)[-1] for path in glob("data/transport/*") if "." not in path[2:]]
    for city in transport_cities:
        datasets[f"transport_{city}"] = Transport(city)

    def __init__(self, dataset_name: str, sample_source: str, sampler_hpars: dict):
        self.seed = None
        self.device = None
        self.val_size = None
        self.test_size = None
        self.dataset_name = dataset_name
        self.community_method = None
        self.leiden_resolution = None
        self.sample_source = sample_source
        self.sampler_hpars = sampler_hpars

        self.dataset: Dataset = None
        self.graph: Graph = None

        self.num_nodes: int = 0
        self.num_node_types: int = 0
        self.num_relations: int = 0
        self.graph_analysis_metrics: dict = {}
        self.relation_freqs: np.ndarray = None
        self.out_degrees: np.ndarray = None
        self.in_degrees: np.ndarray = None
        self.degrees: np.ndarray = None
        self.node_degree_type_freqs: pd.Series = None
        self.node_importances: np.ndarray = None
        self.neighbour_importances: np.ndarray = None
        self.num_communities: int = None
        self.communities: np.ndarray = None
        self.community_sizes: np.ndarray = None
        self.label_community_edge_freqs_index: np.ndarray = None
        self.label_community_edge_freqs: np.ndarray = None
        self.intra_community_map: np.ndarray = None
        self.inter_community_map: np.ndarray = None
        self.com_neighbours: Optional[np.ndarray] = None
        self.node_neighbours: Optional[np.ndarray] = None

        self.graph_indexes: Iterable[AdjacencyIndex] = None
        self.com_adjacency: Optional[Dict[Tuple[int, int], Set[int]]] = None
        self.node_adjacency: Optional[Dict[Tuple[int, int], Set[int]]] = None
        self.train_edge_data: pd.DataFrame = None
        self.val_edge_data: pd.DataFrame = None
        self.test_edge_data: pd.DataFrame = None
        self.sampler: Sampler = SamplerHpars.from_dict(sampler_hpars).make()

    def load_graph(self, seed: int, device: str, val_size: float, test_size: float,
                   community_method: str, leiden_resolution: float):
        print(f"Loading {self.dataset_name} graph...")
        self.seed = seed
        self.device = device
        self.val_size = val_size
        self.test_size = test_size
        self.community_method = community_method
        self.leiden_resolution = leiden_resolution

        self.dataset = Loader.datasets[self.dataset_name]
        self.dataset.load_from_disk()
        self.dataset.time_sort_and_numerize()
        self.graph = Graph(self.dataset_name)
        self.graph.update_graph(self.dataset.node_data, self.dataset.edge_data)
        print("Computing required metrics...")
        self.set_metrics()

        self.train_edge_data, self.val_edge_data, self.test_edge_data = \
            self.dataset.train_val_test_split(self.val_size, self.test_size, self.seed)
        self.graph_indexes = get_efficient_indexes(self.dataset.node_data, self.train_edge_data, self.communities)
        self.sampler.set_neg_filtering_index(self.val_edge_data, self.test_edge_data)

        edge_data_full = self.dataset.edge_data.assign(
            s_type=self.dataset.node_data.type.values[self.dataset.edge_data.s],
            t_type=self.dataset.node_data.type.values[self.dataset.edge_data.t],
            c_s=self.communities[self.dataset.edge_data.s],
            c_t=self.communities[self.dataset.edge_data.t]
        )
        self.label_community_edge_freqs = edge_data_full.groupby(["s_type", "c_s", "r", "c_t", "t_type"]).size()
        community_query_edge_counts = edge_data_full.groupby(["c_s", "r"])["c_t"].value_counts()
        num_edges = len(edge_data_full)
        del edge_data_full
        self.label_community_edge_freqs_index = np.array(list(map(np.array,
                                                                  self.label_community_edge_freqs.index.values))).T
        self.label_community_edge_freqs = self.label_community_edge_freqs.values / self.label_community_edge_freqs.sum()
        hits_at_1_limit = community_query_edge_counts.groupby(["c_s", "r"]).head(1).sum() / num_edges
        hits_at_3_limit = community_query_edge_counts.groupby(["c_s", "r"]).head(3).sum() / num_edges
        hits_at_10_limit = community_query_edge_counts.groupby(["c_s", "r"]).head(10).sum() / num_edges
        community_query_counts = community_query_edge_counts.groupby(["c_s", "r"]).count()
        community_query_edge_counts = community_query_edge_counts.to_frame().assign(rank=0, rrank=0)
        for (c_s, r), c_t_count in community_query_counts.iteritems():
            community_query_edge_counts.loc[(c_s, r), "rank"] = np.arange(1, c_t_count + 1)
            community_query_edge_counts.loc[(c_s, r), "rrank"] = 1 / np.arange(1, c_t_count + 1)
        mr_limit = (community_query_edge_counts["c_t"] * community_query_edge_counts["rank"]).sum() / num_edges
        mrr_limit = (community_query_edge_counts["c_t"] * community_query_edge_counts["rrank"]).sum() / num_edges
        print(f"ComHITS@1 upper bound: {hits_at_1_limit:.4f}")
        print(f"ComHITS@3 upper bound: {hits_at_3_limit:.4f}")
        print(f"ComHITS@10 upper bound: {hits_at_10_limit:.4f}")
        print(f"ComMR lower bound: {mr_limit:.4f}")
        print(f"ComMRR upper bound: {mrr_limit:.4f}")

        _, adj_s_to_t, adj_t_to_s, adj_s_to_t_c, adj_t_to_s_c = self.graph_indexes
        self.com_adjacency = {}
        for c_s, c_s_relations in adj_s_to_t_c.items():
            for r, c_t_list in c_s_relations.items():
                self.com_adjacency.setdefault((c_s, r), set())
                self.com_adjacency[(c_s, r)].update(c_t_list)
        for c_t, c_t_relations in adj_t_to_s_c.items():
            for r, c_s_list in c_t_relations.items():
                self.com_adjacency.setdefault((c_t, r), set())
                self.com_adjacency[(c_t, r)].update(c_s_list)

        self.com_neighbours = np.arange(self.num_communities).repeat(self.num_relations
                                                                     * self.sampler.num_neighbours)
        self.com_neighbours = self.com_neighbours.reshape((self.num_communities,
                                                           self.num_relations,
                                                           self.sampler.num_neighbours))

        for (c, r) in tqdm(self.com_adjacency, "Sampling community neighbours", leave=False):
            self.com_neighbours[c, r] = get_neighbours_entity(self.com_adjacency, c, r, self.sampler.num_neighbours)
        np.savez_compressed(f"graph_completion/results/{self.dataset_name}/"
                            f"community_neighbours_{self.sampler.num_neighbours}.npz",
                            com_neighbours=self.com_neighbours)

        self.node_adjacency = {}
        for s, s_relations in adj_s_to_t.items():
            for r, t_list in s_relations.items():
                self.node_adjacency.setdefault((s, r), set())
                self.node_adjacency[(s, r)].update(t_list)
        for t, t_relations in adj_t_to_s.items():
            for r, s_list in t_relations.items():
                self.node_adjacency.setdefault((t, r), set())
                self.node_adjacency[(t, r)].update(s_list)

        if not isfile(f"graph_completion/results/{self.dataset_name}/"
                      f"node_neighbours_{self.sampler.num_neighbours}.npz"):
            self.node_neighbours = np.arange(self.num_nodes).repeat(self.num_relations * self.sampler.num_neighbours)
            self.node_neighbours = self.node_neighbours.reshape((self.num_nodes,
                                                                 self.num_relations,
                                                                 self.sampler.num_neighbours))

            for (n, r) in tqdm(self.node_adjacency, "Sampling node neighbours", leave=False):
                self.node_neighbours[n, r] = get_neighbours_entity(self.node_adjacency, n, r,
                                                                   self.sampler.num_neighbours)
            np.savez_compressed(f"graph_completion/results/{self.dataset_name}/"
                                f"node_neighbours_{self.sampler.num_neighbours}.npz",
                                node_neighbours=self.node_neighbours)
        else:
            self.node_neighbours = np.load(f"graph_completion/results/{self.dataset_name}/"
                                           f"node_neighbours_{self.sampler.num_neighbours}.npz")["node_neighbours"]

    def set_metrics(self):
        self.num_nodes, self.num_node_types, self.num_relations = \
            len(self.dataset.node_data), len(self.dataset.node_types_map), len(self.dataset.relation_names_map)
        print(f"Num nodes: {self.num_nodes}")
        print(f"Num edges: {len(self.dataset.edge_data)}")
        print(f"Num node types: {self.num_node_types}")
        print(f"Num relations: {self.num_relations}")

        if not isfile(f"graph_completion/results/{self.dataset_name}/heterogeneity.gz"):
            metric_set = Heterogeneity(self.graph, self.num_node_types, self.num_relations)
            metric_set.recursive_updates(self.dataset.node_data, self.dataset.edge_data)
            metric_set.compute_metrics()
            metric_values = metric_set.metric_values
            makedirs(f"graph_completion/results/{self.dataset_name}", exist_ok=True)
            save_object(metric_set.simplified_view(), f"graph_completion/results/{self.dataset_name}/heterogeneity.gz")
        else:
            _, _, metric_values = load_object(f"graph_completion/results/{self.dataset_name}/heterogeneity.gz")
        self.relation_freqs = metric_values["RelationDist"][-1]
        print(f"Edge density: {metric_values['EdgeDensity'][-1]:.8f}")
        print(f"Node type assortativity: {metric_values['NodeTypeAssortativity'][-1]:.4f}")
        self.graph_analysis_metrics["EdgeDensity"] = metric_values['EdgeDensity'][-1]
        self.graph_analysis_metrics["NodeTypeAssortativity"] = metric_values['NodeTypeAssortativity'][-1]

        if not isfile(f"graph_completion/results/{self.dataset_name}/connectivity.gz"):
            metric_set = Connectivity(self.graph, self.num_nodes)
            metric_set.recursive_updates(self.dataset.node_data, self.dataset.edge_data)
            metric_set.compute_metrics()
            metric_values = metric_set.metric_values
            makedirs(f"graph_completion/results/{self.dataset_name}", exist_ok=True)
            save_object(metric_set.simplified_view(), f"graph_completion/results/{self.dataset_name}/connectivity.gz")
        else:
            _, _, metric_values = load_object(f"graph_completion/results/{self.dataset_name}/connectivity.gz")
        self.out_degrees = np.rint(len(self.dataset.edge_data) * metric_values["OutDegree"][-1]).astype(int)
        self.in_degrees = np.rint(len(self.dataset.edge_data) * metric_values["InDegree"][-1]).astype(int)
        self.degrees = self.out_degrees + self.in_degrees
        self.node_degree_type_freqs = pd.concat((pd.Series(self.degrees, name="degree"),
                                                 self.dataset.node_data.type), axis="columns")
        self.node_degree_type_freqs = self.node_degree_type_freqs.value_counts(normalize=True)
        print(f"Largest WCC: {metric_values['GiantWCC'][-1]:.4f}")
        print(f"Largest SCC: {metric_values['GiantSCC'][-1]:.4f}")
        print(f"Average path length: {metric_values['AveragePathLength'][-1]:.4f}")
        print(f"Diameter: {metric_values['Diameter'][-1]}")
        print(f"Average clustering: {metric_values['AverageClustering'][-1]:.6f}")
        self.graph_analysis_metrics["GiantWCC"] = metric_values['GiantWCC'][-1]
        self.graph_analysis_metrics["GiantSCC"] = metric_values['GiantSCC'][-1]
        self.graph_analysis_metrics["AveragePathLength"] = metric_values['AveragePathLength'][-1]
        self.graph_analysis_metrics["Diameter"] = metric_values['Diameter'][-1]
        self.graph_analysis_metrics["AverageClustering"] = metric_values['AverageClustering'][-1]

        if not isfile(f"graph_completion/results/{self.dataset_name}/importance.gz"):
            metric_set = Importance(self.graph, self.num_nodes)
            metric_set.recursive_updates(self.dataset.node_data, self.dataset.edge_data)
            metric_set.compute_metrics()
            metric_values = metric_set.metric_values
            makedirs(f"graph_completion/results/{self.dataset_name}", exist_ok=True)
            save_object(metric_set.simplified_view(), f"graph_completion/results/{self.dataset_name}/importance.gz")
        else:
            _, _, metric_values = load_object(f"graph_completion/results/{self.dataset_name}/importance.gz")
        self.node_importances = metric_values[
            "PageRank" if self.sampler_hpars["pagerank_importances"] else "HubScore"
        ][-1]
        self.neighbour_importances = metric_values[
            "PageRank" if self.sampler_hpars["pagerank_importances"] else "AuthorityScore"
        ][-1]

        metric_set = Subgraphing(self.graph, self.num_nodes, self.community_method, self.leiden_resolution)
        metric_set.recursive_updates(self.dataset.node_data, self.dataset.edge_data)
        metric_set.compute_metrics()
        metric_values = metric_set.metric_values
        makedirs(f"graph_completion/results/{self.dataset_name}", exist_ok=True)
        save_object(metric_set.simplified_view(), f"graph_completion/results/{self.dataset_name}/subgraphing.gz")

        self.communities = metric_values["DisjointCommunityMembership"][-1]
        self.num_communities = int(metric_values['DisjointCommunityNumber'][-1])
        self.community_sizes = np.rint(
            self.num_nodes * metric_values["DisjointCommunitySizeDist"][-1][:self.num_communities]
        ).astype(int)
        self.intra_community_map = np.zeros(self.num_nodes, dtype=int)
        self.inter_community_map = np.zeros(self.num_nodes, dtype=int)
        for community in range(self.num_communities):
            nodes_in_community = self.communities == community
            self.intra_community_map[nodes_in_community] = np.arange(self.community_sizes[community])
        inter_community_edges = self.communities[self.dataset.edge_data.s] != self.communities[self.dataset.edge_data.t]
        inter_community_nodes = np.unique(np.concatenate((self.dataset.edge_data[inter_community_edges].s.values,
                                                          self.dataset.edge_data[inter_community_edges].t.values)))
        del inter_community_edges
        self.inter_community_map[inter_community_nodes] = 1 + np.arange(len(inter_community_nodes))
        print(f"Number of communities: {self.num_communities}")
        print(f"Largest community: {metric_values['GiantDisjointCommunity'][-1]:.4f}")
        print(f"Cut size: {metric_values['DisjointCommunityCutSize'][-1]}")
        print(f"Modularity: {metric_values['DisjointCommunityModularity'][-1]:.4f}")
        print(f"Percentage of inter-community nodes: {len(inter_community_nodes) / self.num_nodes:.4f}")

    def get_training_queries(self, batch_size: int) -> Dict[str, QueryData]:
        node_types = self.dataset.node_data.type.values
        if self.sample_source == "random_walks":
            samples = self.sampler.get_query_samples_random_walks(batch_size, self.graph_indexes,
                                                                  self.num_nodes, self.num_relations,
                                                                  self.communities, self.num_communities)
        else:
            samples = self.sampler.get_query_samples_smore(batch_size, self.graph_indexes,
                                                           self.num_nodes, self.num_relations,
                                                           self.communities, self.num_communities)
        query_data = samples_to_tensors(
            samples, node_types, self.communities, self.num_communities,
            (lambda e, r: self.com_neighbours[e, r]) if self.com_neighbours is not None
            else (lambda e, r: get_neighbours_entity(self.com_adjacency, e, r, self.sampler.num_neighbours)),
            (lambda e, r: self.node_neighbours[e, r]) if self.node_neighbours is not None
            else (lambda e, r: get_neighbours_entity(self.node_adjacency, e, r, self.sampler.num_neighbours)),
            self.num_node_types, self.num_relations, self.device
        )

        return query_data

    def get_evaluation_queries(self, val: bool = False) -> Dict[str, QueryData]:
        node_types = self.dataset.node_data.type.values
        n_of_c, adj_s_to_t, adj_t_to_s, adj_s_to_t_c, adj_t_to_s_c = get_efficient_indexes(self.dataset.node_data,
                                                                                           self.dataset.edge_data,
                                                                                           self.communities)
        edge_data = self.val_edge_data if val else self.test_edge_data
        num_negative_samples = self.sampler.num_negative_samples
        self.sampler.num_negative_samples = 5
        samples = []
        for i, row in tqdm(edge_data.iterrows(), f"Building {'validation' if val else 'test'} set",
                           total=len(edge_data), leave=False):
            s, r, t = row["s"], row["r"], row["t"]
            c_s, c_t = self.communities[s], self.communities[t]
            for query in self.sampler.queries:
                try:
                    qi = next(query.instantiate(adj_t_to_s, self.num_nodes, self.num_relations,
                                                t, (r, s)))
                except StopIteration:
                    continue
                try:
                    qi_c = next(query.instantiate(adj_t_to_s_c, self.num_communities, self.num_relations,
                                                  c_t, (r, c_s)))
                except StopIteration:
                    continue
                qi_tree = qi.map_to_tree(query.query_tree)
                qi_tree_c = qi_c.map_to_tree(query.query_tree)
                if self.sample_source == "random_walks":
                    neg_samples = self.sampler.get_neg_samples_random_walks(qi_tree, query, [], -1,
                                                                            n_of_c, self.communities,
                                                                            self.num_communities, True)
                    neg_samples_c = self.sampler.get_neg_samples_random_walks(qi_tree_c, query, [], -1,
                                                                              n_of_c, self.communities,
                                                                              self.num_communities, False)
                else:
                    node_cut_cache = get_node_cut_cache(qi_tree, query, adj_s_to_t)
                    neg_samples = self.sampler.get_neg_samples_smore(qi_tree, query, node_cut_cache,
                                                                     n_of_c, adj_t_to_s,
                                                                     self.communities, self.num_communities,
                                                                     True)
                    node_cut_cache_c = get_node_cut_cache(qi_tree_c, query, adj_s_to_t_c)
                    neg_samples_c = self.sampler.get_neg_samples_smore(qi_tree_c, query, node_cut_cache_c,
                                                                       n_of_c, adj_t_to_s_c,
                                                                       self.communities, self.num_communities,
                                                                       False)
                if len(neg_samples) == 0:
                    neg_samples = np.random.choice(self.num_nodes,
                                                   size=self.sampler.num_negative_samples)
                if len(neg_samples_c) == 0:
                    neg_samples_c = np.random.choice(self.num_communities,
                                                     size=self.sampler.num_negative_samples)
                samples.append((query, (qi_tree, neg_samples), (qi_tree_c, neg_samples_c)))

        query_data = samples_to_tensors(
            samples, node_types, self.communities, self.num_communities,
            (lambda e, r: self.com_neighbours[e, r]) if self.com_neighbours is not None
            else (lambda e, r: get_neighbours_entity(self.com_adjacency, e, r, self.sampler.num_neighbours)),
            (lambda e, r: self.node_neighbours[e, r]) if self.node_neighbours is not None
            else (lambda e, r: get_neighbours_entity(self.node_adjacency, e, r, self.sampler.num_neighbours)),
            self.num_node_types, self.num_relations, self.device
        )
        self.sampler.num_negative_samples = num_negative_samples

        return query_data


@attr.s
class LoaderHpars(AbstractConf):
    OPTIONS = {"loader": Loader}
    dataset_name = attr.ib(default="transport_rome", validator=in_(list(Loader.datasets.keys())))
    sample_source = attr.ib(default="smore", validator=in_(["random_walks", "smore"]))
    sampler_hpars = attr.ib(factory=SamplerHpars, validator=lambda i, a, v: type(v) is SamplerHpars)
    name = "loader"
