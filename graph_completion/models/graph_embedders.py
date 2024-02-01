"""
Module containing the implementation of the embedding models from related work.
"""
from abc import ABC, abstractmethod
from math import sqrt
from queue import Queue
from typing import Iterable, Optional, Tuple

import attr
import pandas as pd
import torch as pt
from attr.validators import and_, ge, in_, instance_of, le, optional
from torch import nn
from torch.nn.functional import normalize, relu, softmax
from torch_geometric.nn import GAT, GCNConv, MLP

from graph_completion.graphs.preprocess import QueryData
from graph_completion.utils import AbstractConf


class LinearKG(nn.Module):
    def __init__(self, relation_dim: int, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.relation_dim = relation_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias

        self.W = nn.Parameter(nn.init.xavier_uniform_(pt.empty(relation_dim, output_dim, input_dim)))
        if bias:
            self.b = nn.Parameter(nn.init.xavier_uniform_(pt.empty(relation_dim, output_dim)))

    def forward(self, x: pt.Tensor, edge_attr: pt.Tensor) -> pt.Tensor:
        y = pt.einsum("roi,ni,nr->no", self.W, x, edge_attr)
        if self.bias:
            y = y + edge_attr @ self.b
        return y


class MLPKG(nn.Module):
    def __init__(self, num_relations: int, input_dim: int, output_dim: int,
                 hidden_dim: int, num_hidden_layers: int, bias: bool = True):
        super().__init__()
        self.num_relations = num_relations
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.bias = bias

        self.input_to_hidden = LinearKG(num_relations + 1, input_dim, hidden_dim, bias)
        self.hidden_layers = nn.ModuleList([LinearKG(num_relations + 1, hidden_dim, hidden_dim, bias)
                                            for _ in range(num_hidden_layers)])
        self.hidden_to_output = LinearKG(num_relations + 1, hidden_dim, output_dim, bias)

    def forward(self, x: pt.Tensor, edge_attr: pt.Tensor) -> pt.Tensor:
        y = relu(self.input_to_hidden(x, edge_attr))
        for hidden_layer in self.hidden_layers:
            y = relu(hidden_layer(y, edge_attr))
        y = self.hidden_to_output(y, edge_attr)
        return y


class GraphEmbedder(nn.Module, ABC):
    @abstractmethod
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.entity_attr = entity_attr
        self.rel_attr = rel_attr

    @abstractmethod
    def graph_embed(self, query_data: QueryData):
        raise NotImplementedError

    @abstractmethod
    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        raise NotImplementedError

    def forward(self, query_data: QueryData) -> Tuple[pt.Tensor, pt.Tensor]:
        query_batch = query_data.query_tree_batch
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        entity_emb_attr_q = f"{self.entity_attr}_emb_q"
        if entity_emb_attr_q in query_batch.vs.attributes():
            del query_batch.vs[entity_emb_attr_q]
        query_embedding_queue = Queue()
        branching_semaphore = {}
        for query_anchor in query_data.query.query_anchors:
            anchor_emb = query_batch.vs[query_anchor][entity_emb_attr]
            query_embedding_queue.put((query_anchor, anchor_emb))

        while not query_embedding_queue.empty():
            query_node, entity_emb = query_embedding_queue.get()
            query_batch.vs[query_node][entity_emb_attr_q] = entity_emb
            query_parent_edges = query_batch.es[query_batch.incident(query_node, mode="in")]
            if len(query_parent_edges) == 0:
                continue
            query_parent_edge = query_parent_edges[0]
            if query_parent_edge["r"] == "p":
                query_parent = query_parent_edge.source
                entity_relation_emb = query_parent_edge[rel_emb_attr]
                parent_emb = self._relation_project(entity_emb, entity_relation_emb)
                query_embedding_queue.put((query_parent, parent_emb))
            else:
                query_parent = query_parent_edge.source
                branching_children = query_batch.neighbors(query_parent, mode="out")
                branching_semaphore_key = (query_parent_edge["r"], query_parent)
                branching_semaphore.setdefault(branching_semaphore_key, pt.zeros(len(branching_children), dtype=bool))
                branch_index = branching_children.index(query_node)
                branching_semaphore[branching_semaphore_key][branch_index] = True
            if query_embedding_queue.empty():
                for branching_semaphore_key, flags in branching_semaphore.items():
                    branch_type, query_parent = branching_semaphore_key
                    branching_children = query_batch.neighbors(query_parent, mode="out")
                    branching_children_embeddings = query_batch.vs[branching_children][entity_emb_attr]
                    if branch_type == "d":
                        flags[-1] = not flags[-1]
                        branching_children_embeddings[-1] = self._negate(branching_children_embeddings[-1])
                    elif branch_type == "b":
                        flags[0] = not flags[0]
                        branching_children_embeddings[0] = self._negate(branching_children_embeddings[0])
                    parent_emb = self._intersect(branching_children_embeddings)
                    if pt.all(branching_semaphore[branching_semaphore_key]):
                        query_embedding_queue.put((query_parent, parent_emb))
        query_emb = query_batch.vs[query_data.query.query_answer][entity_emb_attr_q]
        answer_emb = query_batch.vs[query_data.query.query_answer][entity_emb_attr]
        return query_emb, answer_emb


class GraphEmbedderMLP(GraphEmbedder):
    """
    Class representing a general MLP embedder. Node embeddings are relation-specific.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 mlp_hidden_dim: int, mlp_num_hidden_layers: int):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.hidden_dim = mlp_hidden_dim
        self.num_hidden_layers = mlp_num_hidden_layers

        self.entity_embeddings = MLPKG(num_relations, num_entities, embedding_dim,
                                       mlp_hidden_dim, mlp_num_hidden_layers)
        self.projection_mlp = MLPKG(num_relations, embedding_dim, embedding_dim,
                                    mlp_hidden_dim, mlp_num_hidden_layers)

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        for query_edge in query_data.query_tree_batch.es:
            query_source, query_target = query_edge.source, query_edge.target
            source = query_data.query_tree_batch.vs[query_source][self.entity_attr]
            target = query_data.query_tree_batch.vs[query_target][self.entity_attr]
            edge_attr = query_edge[self.rel_attr]
            if (entity_emb_attr not in query_data.query_tree_batch.vs[query_source].attributes()
                    or query_data.query_tree_batch.vs[query_source][entity_emb_attr] is None):
                entity_emb = self.__gatne_pipeline(source, edge_attr)
                query_data.query_tree_batch.vs[query_source][entity_emb_attr] = entity_emb
            if (entity_emb_attr not in query_data.query_tree_batch.vs[query_target].attributes()
                    or query_data.query_tree_batch.vs[query_target][entity_emb_attr] is None):
                entity_emb = self.__gatne_pipeline(target, edge_attr)
                query_data.query_tree_batch.vs[query_target][entity_emb_attr] = entity_emb
        query_data.query_tree_batch.es[rel_emb_attr] = query_data.query_tree_batch.es[self.rel_attr]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        return self.projection_mlp(entity_emb, edge_attr_emb)

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        pass

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderTransE(GraphEmbedder):
    """
    Class representing a TransE embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 margin: float):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.margin = margin

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.r_embeddings = nn.Linear(num_relations + 1, embedding_dim, bias=False)
        nn.init.uniform_(self.entity_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.r_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        query_data.query_tree_batch.vs[entity_emb_attr] = [normalize(self.entity_embeddings(e), dim=-1)
                                                           for e in query_data.query_tree_batch.vs[self.entity_attr]]
        query_data.query_tree_batch.es[rel_emb_attr] = [self.r_embeddings(r)
                                                        for r in query_data.query_tree_batch.es[self.rel_attr]]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        return entity_emb + edge_attr_emb

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        pass

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderDistMult(GraphEmbedder):
    """
    Class representing a DistMult embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 margin: float):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.margin = margin

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.r_embeddings = nn.Linear(num_relations + 1, embedding_dim, bias=False)
        nn.init.uniform_(self.entity_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.r_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        query_data.query_tree_batch.vs[entity_emb_attr] = [normalize(self.entity_embeddings(e), dim=-1)
                                                           for e in query_data.query_tree_batch.vs[self.entity_attr]]
        query_data.query_tree_batch.es[rel_emb_attr] = [self.r_embeddings(r)
                                                        for r in query_data.query_tree_batch.es[self.rel_attr]]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        return entity_emb * edge_attr_emb

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        pass

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderComplEx(GraphEmbedder):
    """
    Class representing a ComplEx embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 margin: float):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.margin = margin

        self.entity_embeddings = nn.Embedding(num_entities, 2 * embedding_dim)
        self.r_embeddings = nn.Linear(num_relations + 1, 2 * embedding_dim, bias=False)
        nn.init.uniform_(self.entity_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.r_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        query_data.query_tree_batch.vs[entity_emb_attr] = [self.entity_embeddings(e)
                                                           for e in query_data.query_tree_batch.vs[self.entity_attr]]
        query_data.query_tree_batch.es[rel_emb_attr] = [self.r_embeddings(r)
                                                        for r in query_data.query_tree_batch.es[self.rel_attr]]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        e_s_real, e_s_imag = pt.chunk(entity_emb, 2, dim=-1)
        e_r_real, e_r_imag = pt.chunk(edge_attr_emb, 2, dim=-1)

        e_sr_real = e_s_real * e_r_real - e_s_imag * e_r_imag
        e_sr_imag = e_s_real * e_r_imag + e_s_imag * e_r_real
        return pt.concat([e_sr_real, e_sr_imag], dim=-1)

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        pass

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderRotatE(GraphEmbedder):
    """
    Class representing a RotatE embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 margin: float):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.margin = margin

        self.entity_embeddings = nn.Embedding(num_entities, 2 * embedding_dim)
        self.r_embeddings = nn.Linear(num_relations + 1, embedding_dim, bias=False)
        nn.init.uniform_(self.entity_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.r_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        query_data.query_tree_batch.vs[entity_emb_attr] = [self.entity_embeddings(e)
                                                           for e in query_data.query_tree_batch.vs[self.entity_attr]]
        query_data.query_tree_batch.es[rel_emb_attr] = [
            self.r_embeddings(r) / ((self.margin + 2) / (self.embedding_dim * pt.pi))
            for r in query_data.query_tree_batch.es[self.rel_attr]
        ]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        e_s_real, e_s_imag = pt.chunk(entity_emb, 2, dim=-1)
        e_r_real, e_r_imag = pt.cos(edge_attr_emb), pt.sin(edge_attr_emb)

        e_sr_real = e_s_real * e_r_real - e_s_imag * e_r_imag
        e_sr_imag = e_s_real * e_r_imag + e_s_imag * e_r_real
        return pt.concat([e_sr_real, e_sr_imag], dim=-1)

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        pass

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderGATNE(GraphEmbedder):
    """
    Class representing a GATNE embedder. Node embeddings are relation-specific.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 gatne_neighbour_attr: str, gatne_edge_embedding_dim: int, gatne_attention_dim: int):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.neighbour_attr = gatne_neighbour_attr
        self.edge_embedding_dim = gatne_edge_embedding_dim
        self.attention_dim = gatne_attention_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -1.0, 1.0)
        self.u = nn.Parameter(
            pt.empty(num_entities, num_relations + 1, gatne_edge_embedding_dim).uniform_(-1.0, 1.0)
        )
        emd_std = 1.0 / sqrt(embedding_dim)
        self.W = nn.Parameter(
            pt.fmod(pt.empty(num_relations + 1, gatne_edge_embedding_dim, gatne_attention_dim).normal_(std=emd_std), 2)
        )
        self.w = nn.Parameter(pt.fmod(pt.empty(num_relations + 1, gatne_attention_dim).normal_(std=emd_std), 2))
        self.M = nn.Parameter(
            pt.fmod(pt.empty(num_relations + 1, gatne_edge_embedding_dim, embedding_dim).normal_(std=emd_std), 2)
        )

    def __gatne_pipeline(self, entity: pt.Tensor, edge_attr: pt.Tensor, entity_neighbours: pt.Tensor):
        u_r_neighbours = self.u[entity_neighbours]
        u_r = pt.mean(u_r_neighbours, dim=1)
        W_r = pt.einsum("nr,rua->nua", edge_attr, self.W)
        w_r = edge_attr @ self.w
        M_r = pt.einsum("nr,rue->nue", edge_attr, self.M)
        attention = softmax(pt.einsum("nra,na->nr", pt.tanh(u_r @ W_r), w_r), dim=1)
        return normalize(self.entity_embeddings(entity) + pt.einsum("nue,nru,nr->ne", M_r, u_r, attention), dim=-1)

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        for query_edge in query_data.query_tree_batch.es:
            query_source, query_target = query_edge.source, query_edge.target
            source = query_data.query_tree_batch.vs[query_source][self.entity_attr]
            target = query_data.query_tree_batch.vs[query_target][self.entity_attr]
            edge_attr = query_edge[self.rel_attr]
            source_neighbours, target_neighbours = query_edge[self.neighbour_attr]

            if (entity_emb_attr not in query_data.query_tree_batch.vs[query_source].attributes()
                    or query_data.query_tree_batch.vs[query_source][entity_emb_attr] is None):
                entity_emb = self.__gatne_pipeline(source, edge_attr, source_neighbours)
                query_data.query_tree_batch.vs[query_source][entity_emb_attr] = entity_emb
            if (entity_emb_attr not in query_data.query_tree_batch.vs[query_target].attributes()
                    or query_data.query_tree_batch.vs[query_target][entity_emb_attr] is None):
                entity_emb = self.__gatne_pipeline(target, edge_attr, target_neighbours)
                query_data.query_tree_batch.vs[query_target][entity_emb_attr] = entity_emb
        query_data.query_tree_batch.es[rel_emb_attr] = query_data.query_tree_batch.es[self.rel_attr]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        return entity_emb

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        pass

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderSACN(GraphEmbedder):
    """
    Class representing a SACN embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 sacn_dropout_rate: float,
                 initial_entity_embeddings: Optional[pt.Tensor] = None):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.dropout_rate = sacn_dropout_rate

        self.edge_data: pt.Tensor = None

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim // 2)
        self.r_weights = nn.Embedding(num_relations + 1, 1)
        self.gc1 = GCNConv(embedding_dim // 2, embedding_dim, normalize=False)
        self.gc1_dropout = nn.Dropout(sacn_dropout_rate)
        self.gc2 = GCNConv(embedding_dim, embedding_dim, normalize=False)
        self.gc2_dropout = nn.Dropout(sacn_dropout_rate)
        self.r_embeddings = nn.Linear(num_relations + 1, embedding_dim, bias=False)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.bn4 = nn.BatchNorm1d(embedding_dim)
        with pt.no_grad():
            if initial_entity_embeddings is not None:
                self.entity_embeddings.weight.data = initial_entity_embeddings

        self.x_full: pt.Tensor = None

    def set_edge_data(self, edge_data: pd.DataFrame, device: str):
        self.edge_data = pt.tensor(edge_data[["s", "r", "t"]].values.T, dtype=pt.long, device=device)

    def compute_x_full(self):
        x = self.entity_embeddings(pt.arange(self.num_entities, dtype=pt.long, device=self.edge_data.device))
        edge_index_full = self.edge_data[[0, 2]]
        edge_weight_full = self.r_weights(self.edge_data[1]).squeeze(1)

        x = self.gc1(x=x, edge_index=edge_index_full, edge_weight=edge_weight_full)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = pt.tanh(x)
        x = self.gc1_dropout(x)
        x = self.gc2(x=x, edge_index=edge_index_full, edge_weight=edge_weight_full)
        if x.size(0) > 1:
            x = self.bn4(x)
        x = pt.tanh(x)
        x = self.gc2_dropout(x)

        self.x_full = x

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        if self.x_full is None:
            self.compute_x_full()
        query_data.query_tree_batch.vs[entity_emb_attr] = [self.x_full[e]
                                                           for e in query_data.query_tree_batch.vs[self.entity_attr]]
        query_data.query_tree_batch.es[rel_emb_attr] = [self.r_embeddings(r)
                                                        for r in query_data.query_tree_batch.es[self.rel_attr]]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        return pt.concat((entity_emb, edge_attr_emb), dim=-1)

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        pass

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderKBGAT(GraphEmbedder):
    """
    Class representing a KBGAT embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 kbgat_num_hops: int, kbgat_num_attention_heads: int, kbgat_attention_dim: int,
                 kbgat_negative_slope: float, kbgat_dropout_rate: float,
                 initial_entity_embeddings: Optional[pt.Tensor] = None,
                 initial_r_embeddings: Optional[pt.Tensor] = None):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.num_hops = kbgat_num_hops
        self.num_attention_heads = kbgat_num_attention_heads
        self.attention_dim = kbgat_attention_dim
        self.negative_slope = kbgat_negative_slope
        self.dropout_rate = kbgat_dropout_rate

        self.edge_data: pt.Tensor = None

        self.entity_embeddings_initial = nn.Embedding(num_entities, embedding_dim)
        self.r_embeddings_initial = nn.Embedding(num_relations + 1, embedding_dim)
        self.multi_head_gat = GAT(in_channels=embedding_dim, edge_dim=embedding_dim,
                                  hidden_channels=kbgat_num_attention_heads * kbgat_attention_dim,
                                  out_channels=kbgat_num_attention_heads * embedding_dim, concat=True,
                                  num_layers=kbgat_num_hops, heads=kbgat_num_attention_heads,
                                  act=nn.LeakyReLU(kbgat_negative_slope), dropout=kbgat_dropout_rate)
        self.entity_embeddings_skip = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.r_embeddings = nn.Linear(num_relations + 1, embedding_dim, bias=False)
        with pt.no_grad():
            if initial_entity_embeddings is not None:
                self.entity_embeddings_initial.weight.data = initial_entity_embeddings
            if initial_r_embeddings is not None:
                self.r_embeddings_initial.weight.data = initial_r_embeddings.T
                self.r_embeddings.weight.data = initial_r_embeddings

        self.x_full: pt.Tensor = None

    def set_edge_data(self, edge_data: pd.DataFrame, device: str):
        self.edge_data = pt.tensor(edge_data[["s", "r", "t"]].values.T, dtype=pt.long, device=device)

    def compute_x_full(self):
        x_init = self.entity_embeddings_initial(pt.arange(self.num_entities,
                                                          dtype=pt.long, device=self.edge_data.device))
        edge_index_full = self.edge_data[[0, 2]]
        edge_attr_full = self.r_embeddings_initial(self.edge_data[1])
        x = self.multi_head_gat(x=x_init, edge_index=edge_index_full, edge_attr=edge_attr_full)
        x = pt.mean(pt.stack(pt.chunk(x, self.num_attention_heads, dim=1), dim=0), dim=0)
        x = normalize(x + self.entity_embeddings_skip(x_init), dim=1)

        self.x_full = x

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        if self.x_full is None:
            self.compute_x_full()
        query_data.query_tree_batch.vs[entity_emb_attr] = [self.x_full[e]
                                                           for e in query_data.query_tree_batch.vs[self.entity_attr]]
        query_data.query_tree_batch.es[rel_emb_attr] = [self.r_embeddings(r)
                                                        for r in query_data.query_tree_batch.es[self.rel_attr]]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        return pt.concat((entity_emb, edge_attr_emb), dim=-1)

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        pass

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderGQE(GraphEmbedder):
    """
    Class representing a GQE embedder. Node embeddings are relation-specific.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 margin: float, gqe_mlp_num_hidden_layers: int):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.margin = margin
        self.mlp_num_hidden_layers = gqe_mlp_num_hidden_layers

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        self.R = nn.Parameter(pt.empty(num_relations + 1, embedding_dim, embedding_dim))
        nn.init.uniform_(self.R.data,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        self.deepsets_mlp = MLP([embedding_dim, ] * (gqe_mlp_num_hidden_layers + 2))
        self.intersection_mlp = MLP([embedding_dim, ] * (gqe_mlp_num_hidden_layers + 2))

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        query_data.query_tree_batch.vs[entity_emb_attr] = [normalize(self.entity_embeddings(e), dim=-1)
                                                           for e in query_data.query_tree_batch.vs[self.entity_attr]]
        query_data.query_tree_batch.es[rel_emb_attr] = query_data.query_tree_batch.es[self.rel_attr]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        return pt.einsum("roi,ni,nr->no", self.R, entity_emb, edge_attr_emb)

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        entities_emb = pt.stack(entities_emb, dim=0)
        num_entities = entities_emb.size(0)

        deepsets = self.deepsets_mlp(entities_emb.view(-1, self.embedding_dim))
        deepsets = self.intersection_mlp(pt.mean(
            deepsets.view(num_entities, -1, self.embedding_dim),
            dim=0))
        return deepsets

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderQuery2Box(GraphEmbedder):
    """
    Class representing a Query2Box embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 margin: float, q2b_mlp_num_hidden_layers: int):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.margin = margin
        self.mlp_num_hidden_layers = q2b_mlp_num_hidden_layers

        self.entity_embeddings = nn.Embedding(num_entities, 2 * embedding_dim)
        self.r_embeddings = nn.Linear(num_relations + 1, 2 * embedding_dim, bias=False)
        nn.init.uniform_(self.entity_embeddings.weight[:, :embedding_dim],
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.r_embeddings.weight[:embedding_dim],
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight[:, embedding_dim:],
                         0, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.r_embeddings.weight[embedding_dim:],
                         0, (self.margin + 2) / embedding_dim)
        self.center_mlp = MLP([embedding_dim, ] * (q2b_mlp_num_hidden_layers + 2))
        self.offset_mlp = MLP([embedding_dim, ] * (q2b_mlp_num_hidden_layers + 2))
        self.intersection_mlp = MLP([embedding_dim, ] * (q2b_mlp_num_hidden_layers + 2))

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        query_data.query_tree_batch.vs[entity_emb_attr] = [self.entity_embeddings(e)
                                                           for e in query_data.query_tree_batch.vs[self.entity_attr]]
        query_data.query_tree_batch.es[rel_emb_attr] = [self.r_embeddings(r)
                                                        for r in query_data.query_tree_batch.es[self.rel_attr]]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        return entity_emb + edge_attr_emb

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        entities_emb = pt.stack(entities_emb, dim=0)
        num_entities = entities_emb.size(0)

        center, offset = pt.chunk(entities_emb, 2, dim=-1)
        center = center.view(-1, self.embedding_dim)
        center = self.center_mlp(center).view(num_entities, -1, self.embedding_dim)
        attention = softmax(center, dim=0)
        intersection_center = pt.sum(attention * center, dim=0)

        offset_deepsets = self.offset_mlp(offset.view(-1, self.embedding_dim))
        offset_deepsets = self.intersection_mlp(pt.mean(
            offset_deepsets.view(num_entities, -1, self.embedding_dim),
            dim=0))
        intersection_offset = pt.min(offset, dim=0).values * pt.sigmoid(offset_deepsets)

        return pt.concat((intersection_center, intersection_offset), dim=-1)

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        pass


class GraphEmbedderBetaE(GraphEmbedder):
    """
    Class representing a BetaE embedder. Node embeddings are relation-specific.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, entity_attr: str, rel_attr: str,
                 margin: float, betae_mlp_num_hidden_layers: int):
        super().__init__(num_entities, num_relations, embedding_dim, entity_attr, rel_attr)
        self.margin = margin
        self.mlp_num_hidden_layers = betae_mlp_num_hidden_layers

        self.entity_embeddings = nn.Embedding(num_entities, 2 * embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, 0, (self.margin + 2) / embedding_dim)
        self.relation_mlp = MLPKG(num_relations, 2 * embedding_dim, 2 * embedding_dim,
                                  2 * embedding_dim, betae_mlp_num_hidden_layers)
        self.attention_mlp = MLP([2 * embedding_dim, ] * (betae_mlp_num_hidden_layers + 2))

    def graph_embed(self, query_data: QueryData):
        entity_emb_attr, rel_emb_attr = f"{self.entity_attr}_emb", f"{self.rel_attr}_emb"
        if entity_emb_attr in query_data.query_tree_batch.vs.attributes():
            del query_data.query_tree_batch.vs[entity_emb_attr]
        if rel_emb_attr in query_data.query_tree_batch.es.attributes():
            del query_data.query_tree_batch.es[rel_emb_attr]

        query_data.query_tree_batch.vs[entity_emb_attr] = [pt.clamp(self.entity_embeddings(e) + 1, 0.05, 1e9)
                                                           for e in query_data.query_tree_batch.vs[self.entity_attr]]
        query_data.query_tree_batch.es[rel_emb_attr] = query_data.query_tree_batch.es[self.rel_attr]

    def _relation_project(self, entity_emb: pt.Tensor, edge_attr_emb: pt.Tensor) -> pt.Tensor:
        return pt.clamp(self.relation_mlp(entity_emb, edge_attr_emb) + 1, 0.05, 1e9)

    def _intersect(self, entities_emb: Iterable[pt.Tensor]) -> pt.Tensor:
        entities_emb = pt.stack(entities_emb, dim=0)
        num_entities = entities_emb.size(0)

        attention = self.attention_mlp(entities_emb.view(-1, 2 * self.embedding_dim))
        attention = softmax(attention.view(num_entities, -1, 2 * self.embedding_dim), dim=0)
        return pt.sum(attention * entities_emb, dim=0)

    def _negate(self, entity_emb: pt.Tensor) -> pt.Tensor:
        return 1 / entity_emb


@attr.s
class GraphEmbedderHpars(AbstractConf):
    OPTIONS = {"mlp": GraphEmbedderMLP, "transe": GraphEmbedderTransE, "distmult": GraphEmbedderDistMult,
               "complex": GraphEmbedderComplEx, "rotate": GraphEmbedderRotatE,
               "gatne": GraphEmbedderGATNE, "sacn": GraphEmbedderSACN, "kbgat": GraphEmbedderKBGAT,
               "gqe": GraphEmbedderGQE, "q2b": GraphEmbedderQuery2Box, "betae": GraphEmbedderBetaE}
    algorithm = attr.ib(validator=in_(list(OPTIONS.keys())))
    num_entities = attr.ib(validator=instance_of(int))
    num_relations = attr.ib(validator=instance_of(int))
    entity_attr = attr.ib(validator=instance_of(str))
    rel_attr = attr.ib(validator=instance_of(str))
    margin = attr.ib(validator=instance_of(float))
    embedding_dim = attr.ib(default=25, validator=instance_of(int))
    mlp_hidden_dim = attr.ib(default=128, validator=instance_of(int))
    mlp_num_hidden_layers = attr.ib(default=2, validator=instance_of(int))
    gatne_neighbour_attr = attr.ib(default="n", validator=instance_of(str))
    gatne_edge_embedding_dim = attr.ib(default=10, validator=instance_of(int))
    gatne_attention_dim = attr.ib(default=20, validator=instance_of(int))
    sacn_dropout_rate = attr.ib(default=0.2, validator=and_(instance_of(float), ge(0), le(1)))
    kbgat_num_hops = attr.ib(default=2, validator=instance_of(int))
    kbgat_num_attention_heads = attr.ib(default=2, validator=instance_of(int))
    kbgat_attention_dim = attr.ib(default=100, validator=instance_of(int))
    kbgat_negative_slope = attr.ib(default=0.2, validator=instance_of(float))
    kbgat_dropout_rate = attr.ib(default=0.3, validator=and_(instance_of(float), ge(0), le(1)))
    initial_entity_embeddings = attr.ib(default=None, validator=optional(instance_of(pt.Tensor)))
    initial_r_embeddings = attr.ib(default=None, validator=optional(instance_of(pt.Tensor)))
    gqe_mlp_num_hidden_layers = attr.ib(default=1, validator=instance_of(int))
    q2b_mlp_num_hidden_layers = attr.ib(default=1, validator=instance_of(int))
    betae_mlp_num_hidden_layers = attr.ib(default=1, validator=instance_of(int))

    def __attrs_post_init__(self):
        self.name = self.algorithm
