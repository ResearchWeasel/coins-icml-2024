"""
Module containing the implementation of the new embedding model and its loss function.
"""

from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch as pt
from torch import nn
from torch.nn.functional import normalize, softmax

from graph_completion.graphs.preprocess import QueryData, QueryEmbeddingData
from graph_completion.models.graph_embedders import GraphEmbedderHpars
from graph_completion.models.link_rankers import LinkRankerHpars
from graph_completion.models.loss_terms import EmbeddingLossHpars


class COINs(nn.Module):
    def __init__(self, num_nodes: int, num_node_types: int, num_relations: int,
                 num_communities: int, community_sizes: np.ndarray,
                 intra_community_map: np.ndarray, inter_community_map: np.ndarray,
                 embedder_hpars: dict, transe_model: Optional[dict] = None):
        """
        Construct the embedding model.
        """

        super().__init__()
        self.num_nodes = num_nodes
        self.num_node_types = num_node_types
        self.num_relations = num_relations
        self.num_communities = num_communities
        self.num_inter_community_nodes = int(inter_community_map.max())
        self.node_types = None
        self.community_membership = None
        self.intra_community_map = intra_community_map
        self.inter_community_map = inter_community_map
        self.embedder_hpars = GraphEmbedderHpars.from_dict(embedder_hpars)

        self.algorithm = self.embedder_hpars.algorithm
        self.embedding_dim = self.embedder_hpars.embedding_dim
        self.mlp_num_hidden_layers = self.embedder_hpars.mlp_num_hidden_layers
        self.mlp_hidden_dim = self.embedder_hpars.mlp_hidden_dim

        self.embeddings_complex = self.embedder_hpars.algorithm in ["complex", "rotate", "q2b", "betae"]
        self.embeddings_relation_specific = self.embedder_hpars.algorithm in ["mlp", "gatne"]
        self.embedder_sparse = self.embedder_hpars.algorithm in ["sacn", "kbgat"]

        community_embedder_hpars = deepcopy(self.embedder_hpars)
        community_embedder_hpars.num_entities = num_communities
        community_embedder_hpars.entity_attr = "e_c"
        community_embedder_hpars.rel_attr = "edge_attr_c"
        community_embedder_hpars.gatne_neighbour_attr = "n_c"
        with pt.no_grad():
            if self.embedder_sparse and transe_model is not None:
                community_embedder_hpars.initial_entity_embeddings = \
                    transe_model["community_embedder.entity_embeddings.weight"]
                community_embedder_hpars.initial_r_embeddings = \
                    transe_model["community_embedder.r_embeddings.weight"]
        self.community_embedder = community_embedder_hpars.make()

        self.node_type_embedder = nn.Linear(num_node_types,
                                            2 * self.embedding_dim if self.embeddings_complex else self.embedding_dim,
                                            bias=False)
        if self.algorithm in ["transe", "distmult", "complex", "rotate", "gqe"]:
            margin = self.embedder_hpars.margin
            nn.init.uniform_(self.node_type_embedder.weight,
                             -(margin + 2) / self.embedding_dim, (margin + 2) / self.embedding_dim)
        elif self.algorithm == "q2b":
            margin = self.embedder_hpars.margin
            nn.init.uniform_(self.node_type_embedder.weight[:self.embedding_dim],
                             -(margin + 2) / self.embedding_dim, (margin + 2) / self.embedding_dim)
            nn.init.uniform_(self.node_type_embedder.weight[self.embedding_dim:],
                             0, (margin + 2) / self.embedding_dim)
        elif self.algorithm == "betae":
            margin = self.embedder_hpars.margin
            nn.init.uniform_(self.node_type_embedder.weight,
                             0, (margin + 2) / self.embedding_dim)
        elif self.algorithm == "gatne":
            nn.init.uniform_(self.node_type_embedder.weight, -1.0, 1.0)

        self.intra_community_embedders = []
        for i in range(num_communities):
            intra_community_embedder_hpars = deepcopy(self.embedder_hpars)
            intra_community_embedder_hpars.num_entities = int(community_sizes[i])
            with pt.no_grad():
                if self.embedder_sparse and transe_model is not None:
                    intra_community_embedder_hpars.initial_entity_embeddings = \
                        transe_model[f"intra_community_embedders.{i}.entity_embeddings.weight"]
                    intra_community_embedder_hpars.initial_r_embeddings = \
                        transe_model[f"intra_community_embedders.{i}.r_embeddings.weight"]
            self.intra_community_embedders.append(intra_community_embedder_hpars.make())
        self.intra_community_embedders = nn.ModuleList(self.intra_community_embedders)

        inter_community_embedder_hpars = deepcopy(self.embedder_hpars)
        inter_community_embedder_hpars.num_entities = 1 + self.num_inter_community_nodes
        with pt.no_grad():
            if self.embedder_sparse and transe_model is not None:
                inter_community_embedder_hpars.initial_entity_embeddings = \
                    transe_model["inter_community_embedder.entity_embeddings.weight"]
                inter_community_embedder_hpars.initial_r_embeddings = \
                    transe_model["inter_community_embedder.r_embeddings.weight"]
        self.inter_community_embedder = inter_community_embedder_hpars.make()

        self.final_embeddings_weights = nn.Parameter(pt.ones(3))
        self.final_embeddings_weights_r = nn.Parameter(pt.ones(2))

    def set_graph_data(self, node_types: np.ndarray, edge_data: pd.DataFrame,
                       community_membership: np.ndarray, device: str):
        if self.embedder_sparse:
            self.community_embedder.set_edge_data(
                edge_data.assign(s=community_membership[edge_data.s], t=community_membership[edge_data.t]),
                device
            )

            for i in range(self.num_communities):
                nodes_in_community = community_membership == i
                edge_data_community = edge_data[nodes_in_community[edge_data.s] & nodes_in_community[edge_data.t]]
                self.intra_community_embedders[i].set_edge_data(
                    edge_data_community.assign(s=self.intra_community_map[edge_data_community.s],
                                               t=self.intra_community_map[edge_data_community.t]),
                    device
                )

            self.inter_community_embedder.set_edge_data(
                edge_data.assign(s=self.inter_community_map[edge_data.s], t=self.inter_community_map[edge_data.t]),
                device
            )

        self.node_types = pt.tensor(node_types, dtype=pt.long, device=device)
        self.community_membership = pt.tensor(community_membership, dtype=pt.long, device=device)
        self.intra_community_map = pt.tensor(self.intra_community_map, dtype=pt.long, device=device)
        self.inter_community_map = pt.tensor(self.inter_community_map, dtype=pt.long, device=device)

    def clear_x_full(self):
        if self.embedder_sparse:
            self.community_embedder.x_full = None
            for i in range(self.num_communities):
                self.intra_community_embedders[i].x_full = None
            self.inter_community_embedder.x_full = None

    def embed_communities(self, query_data: QueryData) -> Tuple[pt.Tensor, pt.Tensor]:
        # Community embedding
        self.community_embedder.entity_attr = "e_c"
        self.community_embedder.rel_attr = "edge_attr_c"
        self.community_embedder.gatne_neighbour_attr = "n_c"
        self.community_embedder.graph_embed(query_data)
        return self.community_embedder(query_data)

    def forward(self, query_data: QueryData) -> Tuple[pt.Tensor, pt.Tensor]:

        # Node type embedding
        x_emb = [self.node_type_embedder(x) for x in query_data.query_tree_batch.vs["x"]]
        if self.algorithm == "betae":
            x_emb = [pt.clamp(x + 1, 0.05, 1e9) for x in x_emb]
        query_data.query_tree_batch.vs["x_emb"] = x_emb

        # Community embedding for nodes
        query_data.query_tree_batch.es["c_n"] = [self.community_membership[n]
                                                 for n in query_data.query_tree_batch.es["n"]]
        self.community_embedder.entity_attr = "c"
        self.community_embedder.rel_attr = "edge_attr"
        self.community_embedder.gatne_neighbour_attr = "c_n"
        self.community_embedder.graph_embed(query_data)

        # Node embedding
        query_emb, answer_emb = [], []
        for c_tuple, c_tuple_data in query_data.community_split():

            c_query_unique = pt.unique(c_tuple)
            if len(c_query_unique) == 1:
                # Intra-community case
                node_embedder = self.intra_community_embedders[c_query_unique[0]]
                node_map = self.intra_community_map
            else:
                # Inter-community case
                node_embedder = self.inter_community_embedder
                node_map = self.inter_community_map

            # Final embedding refinement
            c_tuple_data.query_tree_batch.vs["e"] = [node_map[e] for e in c_tuple_data.query_tree_batch.vs["e"]]
            c_tuple_data.query_tree_batch.es["n"] = [node_map[n] for n in c_tuple_data.query_tree_batch.es["n"]]
            node_embedder.graph_embed(c_tuple_data)
            x_emb = c_tuple_data.query_tree_batch.vs["x"]
            c_emb = c_tuple_data.query_tree_batch.vs["e_c"]
            c_edge_attr_emb = c_tuple_data.query_tree_batch.es["edge_attr_c"]
            emb = c_tuple_data.query_tree_batch.vs["e_emb"]
            edge_attr_emb = c_tuple_data.query_tree_batch.es["edge_attr_emb"]
            emb_final = [pt.stack((x, c, e), dim=-1) @ softmax(self.final_embeddings_weights, dim=0)
                         for x, c, e in zip(x_emb, c_emb, emb)]
            edge_attr_emb_final = [pt.stack((c_r, r), dim=-1) @ softmax(self.final_embeddings_weights_r, dim=0)
                                   for c_r, r in zip(c_edge_attr_emb, edge_attr_emb)]

            # Normalization
            if self.algorithm in ["transe", "distmult", "gatne", "kbgat", "gqe"]:
                emb_final = [normalize(e, dim=-1) for e in emb_final]

            c_tuple_data.query_tree_batch.vs["e_emb"] = emb_final
            c_tuple_data.query_tree_batch.es["edge_attr_emb"] = edge_attr_emb_final

            c_tuple_query_emb, c_tuple_answer_emb = node_embedder(c_tuple_data)
            query_emb.append(c_tuple_query_emb)
            answer_emb.append(c_tuple_answer_emb)
        query_emb, answer_emb = pt.cat(query_emb, dim=0), pt.cat(answer_emb, dim=0)

        return query_emb, answer_emb

    def embed_supervised(self, query_data: QueryData) -> QueryEmbeddingData:
        c_query_emb, c_answer_emb = self.embed_communities(query_data)
        query_emb, answer_emb = self(query_data)
        return QueryEmbeddingData(c_query_emb=c_query_emb, c_answer_emb=c_answer_emb,
                                  query_emb=query_emb, answer_emb=answer_emb,
                                  y=query_data.query_tree_batch["y"], sample=query_data.query_tree_batch["sample"])


class COINsLinkPredictor(nn.Module):
    def __init__(self, link_ranker_hpars: dict):
        """
        Construct the link prediction model.
        """

        super().__init__()
        self.link_ranker_hpars = LinkRankerHpars.from_dict(link_ranker_hpars)
        self.algorithm = link_ranker_hpars["algorithm"]

        self.community_link_ranker = self.link_ranker_hpars.make()
        self.node_link_ranker = self.link_ranker_hpars.make()

    def forward(self, query_emb: pt.Tensor, answer_emb: pt.Tensor, for_communities: bool = False) -> pt.Tensor:
        # Query scoring
        link_ranker = self.community_link_ranker if for_communities else self.node_link_ranker
        y = link_ranker(query_emb, answer_emb)

        if self.algorithm != "sacn":
            y = pt.sigmoid(y)

        return y


class COINsLoss(nn.Module):
    def __init__(self, embedding_loss_hpars: dict, coins_alpha: float):
        """
        Construct the loss function for the embedding model.
        """

        super().__init__()
        self.embedding_loss_hpars = EmbeddingLossHpars.from_dict(embedding_loss_hpars)
        self.alpha = coins_alpha
        self.algorithm = embedding_loss_hpars["algorithm"]
        self.learnable_link_ranker = embedding_loss_hpars["algorithm"] in ["mlp", "sacn", "kbgat"]

        self.community_embedding_loss = self.embedding_loss_hpars.make()
        self.embedding_loss = self.embedding_loss_hpars.make()

    def forward(self, query_emb_batch: QueryEmbeddingData, link_ranker: COINsLinkPredictor):
        (c_query, c_query_neg), (c_answer, c_answer_neg), (query, query_neg), (answer, answer_neg) = \
            query_emb_batch.sample_split_embeddings()

        if self.learnable_link_ranker:
            community_loss = self.community_embedding_loss(c_query, c_query_neg, c_answer, c_answer_neg,
                                                           link_ranker.community_link_ranker)
            node_loss = self.embedding_loss(query, query_neg, answer, answer_neg, link_ranker.node_link_ranker)
        else:
            community_loss = self.community_embedding_loss(c_query, c_query_neg, c_answer, c_answer_neg)
            node_loss = self.embedding_loss(query, query_neg, answer, answer_neg)

        return (1 - self.alpha) * community_loss + self.alpha * node_loss, (community_loss, node_loss)
