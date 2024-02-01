from abc import ABC, abstractmethod
from math import sqrt
from typing import Dict, List, Tuple, Union

import igraph
import numpy as np
import pandas as pd
from pymetis import part_graph
from scipy.linalg import eig
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

MetricSetValues = Dict[str, List[Union[float, np.ndarray]]]
MetricSetView = Tuple[str, str, MetricSetValues]


class Graph:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.graph: igraph.Graph = igraph.Graph(directed=True, vertex_attrs={"type": []}, edge_attrs={"r": []})

    def update_graph(self, node_batch: pd.DataFrame, edge_batch: pd.DataFrame):
        self.graph.add_vertices(len(node_batch), {"type": node_batch.type.values})
        self.graph.add_edges(edge_batch[["s", "t"]].itertuples(index=False, name=None), {"r": edge_batch.r.values})


class MetricSet(ABC):
    @abstractmethod
    def __init__(self, graph: Graph, metric_type: str, *args, **kwargs):
        self.graph = graph
        self.metric_type = metric_type
        self.metric_values: MetricSetValues = {}
        self.metric_is_recursive: Dict[str, bool] = {}
        self.current_step = 0
        pass

    def simplified_view(self) -> MetricSetView:
        return self.graph.dataset_name, self.metric_type, self.metric_values

    def __repr__(self) -> str:
        return repr(self.simplified_view())

    @abstractmethod
    def recursive_updates(self, node_batch: pd.DataFrame, edge_batch: pd.DataFrame):
        pass

    @abstractmethod
    def compute_metric(self, metric_name) -> np.ndarray:
        pass

    def compute_metrics(self):
        for metric_name, is_recursive in self.metric_is_recursive.items():
            if not is_recursive:
                self.metric_values[metric_name].append(self.compute_metric(metric_name))
        self.current_step += 1


class Heterogeneity(MetricSet):
    def __init__(self, graph, num_type, num_r):
        super().__init__(graph, "Heterogeneity")
        self.metric_values = {
            "NumNodes": [0, ],
            "NumEdges": [0, ],
            "EdgeDensity": [0, ],
            "NodeTypeDist": [np.zeros(num_type), ],
            "RelationDist": [np.zeros(num_r), ],
            "LabelInteractionDist": [np.zeros((num_type, num_r, num_type)), ],
            "NodeTypeAssortativity": [0, ],
        }
        self.metric_is_recursive = {
            "NumNodes": True,
            "NumEdges": True,
            "EdgeDensity": True,
            "NodeTypeDist": True,
            "RelationDist": False,
            "LabelInteractionDist": True,
            "NodeTypeAssortativity": False,
        }
        self.type_index = pd.Index(range(num_type), name="type")
        self.label_index = pd.MultiIndex.from_product((range(num_type), range(num_r), range(num_type)),
                                                      names=("s_type", "r", "t_type"))

    def recursive_updates(self, node_batch: pd.DataFrame, edge_batch: pd.DataFrame):
        prev_num_nodes = self.metric_values["NumNodes"][self.current_step]
        prev_num_edges = self.metric_values["NumEdges"][self.current_step]
        prev_type_dist = self.metric_values["NodeTypeDist"][self.current_step]
        prev_label_dist = self.metric_values["LabelInteractionDist"][self.current_step]

        self.metric_values["NumNodes"].append(prev_num_nodes + len(node_batch))
        self.metric_values["NumEdges"].append(prev_num_edges + len(edge_batch))
        self.metric_values["EdgeDensity"].append((prev_num_edges + len(edge_batch))
                                                 / (prev_num_nodes + len(node_batch)) ** 2)
        type_counts = prev_type_dist * prev_num_nodes + node_batch.type.value_counts().reindex(self.type_index,
                                                                                               fill_value=0).values
        self.metric_values["NodeTypeDist"].append(type_counts / (prev_num_nodes + len(node_batch)))
        label_batch = edge_batch.assign(
            s_type=self.graph.graph.vs(edge_batch.s)["type"],
            t_type=self.graph.graph.vs(edge_batch.t)["type"],
        )[["s_type", "r", "t_type"]]
        new_label_counts = label_batch.value_counts().reindex(self.label_index,
                                                              fill_value=0).values.reshape(prev_label_dist.shape)
        label_counts = prev_label_dist * prev_num_edges + new_label_counts
        self.metric_values["LabelInteractionDist"].append(label_counts / (prev_num_edges + len(edge_batch)))

    def compute_metric(self, metric_name):
        if metric_name == "RelationDist":
            return np.nansum(self.metric_values["LabelInteractionDist"][self.current_step + 1], axis=(0, 2))
        elif metric_name == "NodeTypeAssortativity":
            return self.graph.graph.assortativity_nominal(types="type",
                                                          directed=True) if len(self.type_index) > 1 else 1


class Connectivity(MetricSet):
    def __init__(self, graph, num_nodes):
        super().__init__(graph, "Connectivity")
        self.metric_values = {
            "OutDegree": [np.zeros(num_nodes), ],
            "InDegree": [np.zeros(num_nodes), ],
            "Degree": [np.zeros(num_nodes), ],
            "WCC": [np.zeros(num_nodes), ],
            "SCC": [np.zeros(num_nodes), ],
            "GiantWCC": [0, ],
            "GiantSCC": [0, ],
            "AveragePathLength": [0, ],
            "Diameter": [0, ],
            "Clustering": [np.zeros(num_nodes), ],
            "AverageClustering": [0, ],
            "DegreeAssortativity": [0, ],
            "AlgebraicConnectivity": [0, ],
        }
        self.metric_is_recursive = {
            "OutDegree": True,
            "InDegree": True,
            "Degree": True,
            "WCC": True,
            "SCC": True,
            "GiantWCC": False,
            "GiantSCC": False,
            "AveragePathLength": False,
            "Diameter": False,
            "Clustering": False,
            "AverageClustering": False,
            "DegreeAssortativity": False,
            "AlgebraicConnectivity": False,
        }
        self.node_index = pd.Index(range(num_nodes), name="n")

    def recursive_updates(self, node_batch: pd.DataFrame, edge_batch: pd.DataFrame):
        prev_e_count = self.graph.graph.ecount()
        prev_out_degree = self.metric_values["OutDegree"][self.current_step]
        prev_in_degree = self.metric_values["InDegree"][self.current_step]

        s_counts = edge_batch.s.value_counts().reindex(self.node_index, fill_value=0).values
        new_s_counts = prev_e_count * prev_out_degree + s_counts
        self.metric_values["OutDegree"].append(new_s_counts / self.graph.graph.ecount())

        t_counts = edge_batch.t.value_counts().reindex(self.node_index, fill_value=0).values
        new_t_counts = prev_e_count * prev_in_degree + t_counts
        self.metric_values["InDegree"].append(new_t_counts / self.graph.graph.ecount())

        self.metric_values["Degree"].append((new_s_counts + new_t_counts) / (2 * self.graph.graph.ecount()))

        wcc = self.graph.graph.clusters(mode="weak").sizes()
        wcc_padded = np.zeros(len(self.node_index))
        wcc_padded[:len(wcc)] = wcc / np.nansum(wcc)
        self.metric_values["WCC"].append(wcc_padded)

        scc = self.graph.graph.clusters(mode="strong").sizes()
        scc_padded = np.zeros(len(self.node_index))
        scc_padded[:len(scc)] = scc / np.nansum(scc)
        self.metric_values["SCC"].append(scc_padded)

    def compute_metric(self, metric_name):
        if metric_name == "GiantWCC":
            return np.nanmax(self.metric_values["WCC"][self.current_step + 1])
        elif metric_name == "GiantSCC":
            return np.nanmax(self.metric_values["SCC"][self.current_step + 1])
        elif metric_name == "AveragePathLength":
            return self.graph.graph.average_path_length(directed=True, unconn=True)
        elif metric_name == "Diameter":
            return self.graph.graph.diameter(directed=True, unconn=True)
        elif metric_name == "Clustering":
            clustering = self.graph.graph.transitivity_local_undirected(mode="zero")
            clustering_padded = np.zeros(len(self.node_index))
            clustering_norm = np.nansum(clustering)
            clustering_padded[:len(clustering)] = (clustering / clustering_norm) if clustering_norm > 0 else clustering
            return clustering_padded
        elif metric_name == "AverageClustering":
            return self.graph.graph.transitivity_undirected(mode="zero")
        elif metric_name == "DegreeAssortativity":
            return self.graph.graph.assortativity_degree(directed=True)
        elif metric_name == "AlgebraicConnectivity":
            A = self.graph.graph.clusters("weak").giant()
            num_nodes = A.vcount()
            D_ind = np.arange(num_nodes)
            D = A.degree()
            D = csr_matrix((D, (D_ind, D_ind)))
            A = A.get_adjacency_sparse()
            L = D - A
            if num_nodes > 3:
                return abs(eigs(L.asfptype(), k=2, which="SM", return_eigenvectors=False, tol=1e-4)[0])
            else:
                eigvals = np.abs(eig(L.asfptype().toarray("C"))[0])
                eigvals = np.sort(eigvals)
                return eigvals[1] if num_nodes > 1 else 0


class Importance(MetricSet):
    def __init__(self, graph, num_nodes):
        super().__init__(graph, "Importance")
        self.metric_values = {
            "ClosenessCentrality": [np.zeros(num_nodes), ],
            "BetweennessCentrality": [np.zeros(num_nodes), ],
            "HubScore": [np.zeros(num_nodes), ],
            "AuthorityScore": [np.zeros(num_nodes), ],
            "PageRank": [np.zeros(num_nodes), ],
            "MaxCloseness": [0, ],
            "MaxBetweenness": [0, ],
            "MaxHubScore": [0, ],
            "MaxAuthorityScore": [0, ],
            "MaxPageRank": [0, ],
        }
        self.metric_is_recursive = {
            "ClosenessCentrality": True,
            "BetweennessCentrality": True,
            "HubScore": True,
            "AuthorityScore": True,
            "PageRank": True,
            "MaxCloseness": False,
            "MaxBetweenness": False,
            "MaxHubScore": False,
            "MaxAuthorityScore": False,
            "MaxPageRank": False,
        }
        self.num_nodes = num_nodes

    def recursive_updates(self, node_batch: pd.DataFrame, edge_batch: pd.DataFrame):
        closeness = self.graph.graph.harmonic_centrality(mode="all", cutoff=10)
        closeness_padded = np.zeros(self.num_nodes)
        closeness_norm = np.nansum(closeness)
        closeness_padded[:len(closeness)] = \
            (closeness / closeness_norm) if closeness_norm > 0 else closeness
        self.metric_values["ClosenessCentrality"].append(closeness_padded)

        betweenness = self.graph.graph.betweenness(directed=True, cutoff=10)
        betweenness_padded = np.zeros(self.num_nodes)
        betweenness_norm = np.nansum(betweenness)
        betweenness_padded[:len(betweenness)] = \
            (betweenness / betweenness_norm) if betweenness_norm > 0 else betweenness
        self.metric_values["BetweennessCentrality"].append(betweenness_padded)

        hub_score = self.graph.graph.hub_score()
        hub_score_padded = np.zeros(self.num_nodes)
        hub_score_padded[:len(hub_score)] = hub_score / np.nansum(hub_score)
        self.metric_values["HubScore"].append(hub_score_padded)

        authority_score = self.graph.graph.authority_score()
        authority_score_padded = np.zeros(self.num_nodes)
        authority_score_padded[:len(authority_score)] = authority_score / np.nansum(authority_score)
        self.metric_values["AuthorityScore"].append(authority_score_padded)

        pagerank = self.graph.graph.pagerank(directed=True)
        pagerank_padded = np.zeros(self.num_nodes)
        pagerank_padded[:len(pagerank)] = pagerank / np.nansum(pagerank)
        self.metric_values["PageRank"].append(pagerank_padded)

    def compute_metric(self, metric_name):
        if metric_name == "MaxCloseness":
            return np.nanmax(self.metric_values["ClosenessCentrality"][self.current_step + 1])
        elif metric_name == "MaxBetweenness":
            return np.nanmax(self.metric_values["BetweennessCentrality"][self.current_step + 1])
        elif metric_name == "MaxHubScore":
            return np.nanmax(self.metric_values["HubScore"][self.current_step + 1])
        elif metric_name == "MaxAuthorityScore":
            return np.nanmax(self.metric_values["AuthorityScore"][self.current_step + 1])
        elif metric_name == "MaxPageRank":
            return np.nanmax(self.metric_values["PageRank"][self.current_step + 1])


def get_sp_tree(graph: igraph.Graph) -> igraph.Graph:
    pageranks_nodes = np.array(graph.pagerank(directed=True))
    edge_frame = graph.get_edge_dataframe()
    graph.es["weight"] = -pageranks_nodes[edge_frame.source.values] - pageranks_nodes[edge_frame.target.values]
    del edge_frame, pageranks_nodes
    sp_tree = graph.spanning_tree()
    del graph.es["weight"]
    return sp_tree


class Subgraphing(MetricSet):
    def __init__(self, graph, num_nodes, community_method, community_resolution_disjoint):
        super().__init__(graph, "Subgraphing")
        self.num_nodes = num_nodes
        self.community_method = community_method
        self.community_resolution_disjoint = \
            community_resolution_disjoint if community_resolution_disjoint > 0 else 1 / num_nodes
        self.node_index = pd.Index(range(num_nodes), name="n")

        self.metric_values = {
            "DisjointCommunityMembership": [np.zeros(num_nodes), ],
            "DisjointCommunityNumber": [0, ],
            "DisjointCommunitySizeDist": [np.zeros(num_nodes), ],
            "GiantDisjointCommunity": [0, ],
            "DisjointCommunityCutSize": [0, ],
            "DisjointCommunityModularity": [0, ],
        }
        self.metric_is_recursive = {
            "DisjointCommunityMembership": True,
            "DisjointCommunityNumber": False,
            "DisjointCommunitySizeDist": True,
            "GiantDisjointCommunity": False,
            "DisjointCommunityCutSize": True,
            "DisjointCommunityModularity": False,
        }

    def recursive_updates(self, node_batch: pd.DataFrame, edge_batch: pd.DataFrame):
        if self.community_method != "random":
            disjoint_partitioning = np.zeros(self.num_nodes, dtype=int)
            wcc = self.graph.graph.clusters("weak")
            wcc_membership, wcc_sizes = np.array(wcc.membership), wcc.sizes()
            nodes_in_larger_wcc = np.zeros(self.num_nodes, dtype=bool)
            small_wcc_community, small_wcc_nodes = 0, 0
            for i, wcc_size in enumerate(wcc_sizes):
                nodes_in_wcc = wcc_membership == i
                if wcc_size < sqrt(self.num_nodes):
                    if small_wcc_nodes + wcc_size > sqrt(self.num_nodes):
                        small_wcc_community += 1
                        small_wcc_nodes = 0
                    disjoint_partitioning[nodes_in_wcc] = small_wcc_community
                    small_wcc_nodes += wcc_size
                else:
                    nodes_in_larger_wcc[nodes_in_wcc] = True
            nodes_in_larger_wcc = nodes_in_larger_wcc.nonzero()[0]
        if self.community_method == "leiden":
            if len(nodes_in_larger_wcc) != self.num_nodes:
                graph_larger_wcc = self.graph.graph.subgraph(nodes_in_larger_wcc)
                larger_wcc_comms = graph_larger_wcc.as_undirected(mode="each").community_leiden(
                    "CPM", resolution_parameter=self.community_resolution_disjoint,
                    n_iterations=-1 if self.num_nodes <= 100000 else 2
                )
                disjoint_partitioning[nodes_in_larger_wcc] = (np.array(larger_wcc_comms.membership)
                                                              + small_wcc_community + 1)
                del graph_larger_wcc
            else:
                disjoint_partitioning = self.graph.graph.as_undirected(mode="each").community_leiden(
                    "CPM", resolution_parameter=self.community_resolution_disjoint,
                    n_iterations=-1 if self.num_nodes <= 100000 else 2
                )
                disjoint_partitioning = np.array(disjoint_partitioning.membership)
            edges = self.graph.graph.get_edge_dataframe()
            cut_size = np.sum(disjoint_partitioning[edges.source] != disjoint_partitioning[edges.target])
            self.metric_values["DisjointCommunityCutSize"].append(cut_size)
            del wcc, wcc_membership, wcc_sizes, nodes_in_larger_wcc
        elif self.community_method == "metis":
            if len(nodes_in_larger_wcc) != self.num_nodes:
                graph_larger_wcc = self.graph.graph.subgraph(nodes_in_larger_wcc)
                graph_metis = graph_larger_wcc.get_adjlist(mode="all")
                cut_size, larger_wcc_comms = part_graph(adjacency=graph_metis,
                                                        nparts=int(sqrt(self.num_nodes)) - small_wcc_community)
                disjoint_partitioning[nodes_in_larger_wcc] = np.array(larger_wcc_comms) + small_wcc_community + 1
                del graph_larger_wcc
            else:
                graph_metis = self.graph.graph.get_adjlist(mode="all")
                cut_size, disjoint_partitioning = part_graph(adjacency=graph_metis, nparts=int(sqrt(self.num_nodes)))
                disjoint_partitioning = np.array(disjoint_partitioning)
            self.metric_values["DisjointCommunityCutSize"].append(cut_size)
            del wcc, wcc_membership, wcc_sizes, nodes_in_larger_wcc
        elif self.community_method == "random":
            disjoint_partitioning = np.random.choice(int(sqrt(self.num_nodes)), size=self.num_nodes)
            edges = self.graph.graph.get_edge_dataframe()
            cut_size = np.sum(disjoint_partitioning[edges.source] != disjoint_partitioning[edges.target])
            self.metric_values["DisjointCommunityCutSize"].append(cut_size)
        else:
            raise ValueError("Invalid community detection method!")

        self.metric_values["DisjointCommunityMembership"].append(disjoint_partitioning)
        disjoint_sizes = np.unique(disjoint_partitioning, return_counts=True)[1]
        disjoint_sizes_padded = np.zeros(self.num_nodes)
        disjoint_sizes_padded[:len(disjoint_sizes)] = disjoint_sizes / np.nansum(disjoint_sizes)
        self.metric_values["DisjointCommunitySizeDist"].append(disjoint_sizes_padded)

    def compute_metric(self, metric_name):
        if metric_name == "DisjointCommunityNumber":
            return np.nanmax(self.metric_values["DisjointCommunityMembership"][self.current_step + 1]) + 1
        elif metric_name == "GiantDisjointCommunity":
            return np.nanmax(self.metric_values["DisjointCommunitySizeDist"][self.current_step + 1])
        elif metric_name == "DisjointCommunityModularity":
            return self.graph.graph.modularity(
                self.metric_values["DisjointCommunityMembership"][self.current_step + 1]
            )
