"""
Module containing the implementation of the evaluation procedure and the experiment pipeline.
"""
import time
from copy import deepcopy
from glob import glob
from math import ceil
from os import makedirs
from os.path import dirname, exists, realpath
from typing import Dict, Optional, Tuple

import attr
import numpy as np
import pandas as pd
import torch as pt
from attr.validators import and_, ge, in_, instance_of, le
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, \
    precision_score, recall_score, roc_auc_score
from tensorboardX import SummaryWriter
from torch.cuda import device_count
from torch.nn.functional import one_hot
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from graph_completion.graphs.load_graph import Loader, LoaderHpars
from graph_completion.graphs.preprocess import get_efficient_indexes, QueryData
from graph_completion.graphs.queries import get_all_answers, query_edge_r_to_int
from graph_completion.models.coins import COINs, COINsLinkPredictor, COINsLoss
from graph_completion.models.graph_embedders import GraphEmbedderHpars
from graph_completion.models.link_rankers import LinkRankerHpars
from graph_completion.models.loss_terms import EmbeddingLossHpars
from graph_completion.utils import AbstractConf, reproduce


class Experiment:
    def __init__(self, seed: int, device: str, train: bool, checkpoint_run: int, checkpoint_tag: str,
                 val_size: float, test_size: float, mini_batch_size: int,
                 lr: float, weight_decay: float, val_patience: int, val_tolerance: float,
                 max_epochs: int, validation_freq: int, checkpoint_freq: int, algorithm: str, community_method: str,
                 leiden_resolution: float, coins_alpha: float, transe_initialize: bool,
                 loader_hpars: dict, embedder_hpars: dict, link_ranker_hpars: dict, embedding_loss_hpars: dict):
        self.seed = seed
        self.device = device
        self.train = train
        self.checkpoint_run = checkpoint_run
        self.checkpoint_tag = checkpoint_tag
        self.val_size = val_size
        self.test_size = test_size
        self.mini_batch_size = mini_batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_patience = val_patience
        self.val_tolerance = val_tolerance
        self.max_epochs = max_epochs
        self.validation_freq = validation_freq
        self.checkpoint_freq = checkpoint_freq
        self.algorithm = algorithm
        self.community_method = community_method
        self.leiden_resolution = leiden_resolution
        self.coins_alpha = coins_alpha
        self.transe_initialize = transe_initialize
        self.loader_hpars = loader_hpars
        self.embedder_hpars = embedder_hpars
        self.link_ranker_hpars = link_ranker_hpars
        self.embedding_loss_hpars = embedding_loss_hpars

        self.loader: Loader = None
        self.val_set: Dict[str, QueryData] = None
        self.test_set: Dict[str, QueryData] = None
        self.embedder: COINs = None
        self.link_ranker: COINsLinkPredictor = None
        self.criterion: COINsLoss = None
        self.embedder_optim: Optimizer = None
        self.link_ranker_optim: Optimizer = None

        self.run_id: int = 0
        self.dashboard: SummaryWriter = None

    def prepare(self):
        reproduce(self.seed)

        # Load data
        self.loader = LoaderHpars.from_dict(self.loader_hpars).make()
        self.loader.load_graph(self.seed, self.device, self.val_size, self.test_size,
                               self.community_method, self.leiden_resolution)
        self.val_set = self.loader.get_evaluation_queries(val=True)
        self.test_set = self.loader.get_evaluation_queries(val=False)

        # Initialize models
        print("Constructing embedder...")
        self.embedder_hpars.update(num_entities=self.loader.num_nodes, num_relations=self.loader.num_relations)
        self.embedder_hpars.update(entity_attr="e", rel_attr="edge_attr")
        self.embedder_hpars.update(margin=self.embedding_loss_hpars["margin"])
        transe_model = None
        if self.transe_initialize and exists(f"graph_completion/results/{self.loader.dataset_name}/transe_model.tar"):
            transe_model = pt.load(f"graph_completion/results/{self.loader.dataset_name}/transe_model.tar",
                                   map_location=self.device)
        self.embedder = COINs(self.loader.num_nodes, self.loader.num_node_types, self.loader.num_nodes,
                              self.loader.num_communities, self.loader.community_sizes,
                              self.loader.intra_community_map, self.loader.inter_community_map,
                              self.embedder_hpars, transe_model).to(self.device)
        self.embedder.set_graph_data(self.loader.dataset.node_data.type.values, self.loader.train_edge_data,
                                     self.loader.communities, self.device)
        self.embedder_optim = Adam(self.embedder.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        print("Constructing link ranker and loss...")
        self.link_ranker_hpars.update(embedding_dim=self.embedder.embedding_dim)
        if self.algorithm in ["transe", "rotate"]:
            self.link_ranker_hpars.update(margin=self.embedding_loss_hpars["margin"])
        self.link_ranker = COINsLinkPredictor(self.link_ranker_hpars).to(self.device)
        self.criterion = COINsLoss(self.embedding_loss_hpars, self.coins_alpha).to(self.device)
        if self.criterion.learnable_link_ranker:
            self.link_ranker_optim = Adam(self.link_ranker.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Logging
        makedirs(f"graph_completion/results/{self.loader.dataset_name}/runs", exist_ok=True)
        self.run_id = len(glob(f"graph_completion/results/{self.loader.dataset_name}/runs/*")) + 1
        makedirs(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}", exist_ok=True)
        self.dashboard = SummaryWriter(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}")

    def run_training(self):
        makedirs(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}/checkpoints", exist_ok=True)
        if self.train and self.checkpoint_run > 0 and len(self.checkpoint_tag) > 0:
            checkpoint = pt.load(f"graph_completion/results/{self.loader.dataset_name}/runs/"
                                 f"{self.checkpoint_run}/checkpoints/checkpoint_{self.checkpoint_tag}.tar",
                                 map_location=self.device)
            start_time = checkpoint["start_time"]
            prev_train_loss, best_val_loss = checkpoint["prev_train_loss"], checkpoint["best_val_loss"]
            num_samples_processed = checkpoint["num_samples_processed"]
            num_batches_processed = checkpoint["num_batches_processed"]
            patience = checkpoint["patience"]
            self.loader.sampler.query_index = checkpoint["query_index"]
            self.loader.sampler.answer_indices = checkpoint["answer_indices"]
            self.loader.sampler.answer_indices_c = checkpoint["answer_indices_c"]
            self.embedder.load_state_dict(checkpoint["embedder_state_dict"])
            self.link_ranker.load_state_dict(checkpoint["link_ranker_state_dict"])
            self.embedder_optim.load_state_dict(checkpoint["embedder_optim_state_dict"])
            if self.criterion.learnable_link_ranker:
                self.link_ranker_optim.load_state_dict(checkpoint["link_ranker_optim_state_dict"])
        else:
            start_time = time.time()
            prev_train_loss, best_val_loss = 0, None
            num_samples_processed = 0
            num_batches_processed = 0
            patience = 0
        best_embedder, best_link_ranker = deepcopy(self.embedder.state_dict()), deepcopy(self.link_ranker.state_dict())

        print("Training...")
        epochs_to_samples = len(self.loader.train_edge_data) * (1 + self.loader.sampler.num_negative_samples)
        while (self.train and patience < self.val_patience
               and num_samples_processed < self.max_epochs * epochs_to_samples):
            self.embedder.train()
            self.embedder_optim.zero_grad(set_to_none=True)
            if self.criterion.learnable_link_ranker:
                self.link_ranker.train()
                self.link_ranker_optim.zero_grad(set_to_none=True)
            self.embedder.clear_x_full()

            num_queries = 0
            loss, com_loss, node_loss = pt.zeros(3, device=self.device)
            train_batch_dict = self.loader.get_training_queries(self.mini_batch_size)
            for _, train_batch in train_batch_dict.items():
                num_queries += 1
                loss_q, (com_loss_q, node_loss_q) = self.criterion(self.embedder.embed_supervised(train_batch),
                                                                   self.link_ranker)
                loss = loss + loss_q
                com_loss = com_loss + com_loss_q
                node_loss = node_loss + node_loss_q
            loss, com_loss, node_loss = loss / num_queries, com_loss / num_queries, node_loss / num_queries
            loss.backward()
            self.embedder_optim.step()
            if self.criterion.learnable_link_ranker:
                self.link_ranker_optim.step()
            self.dashboard.add_scalar("Train/Loss", loss.item(), num_samples_processed)
            self.dashboard.add_scalar("Train/ComLoss", com_loss.item(), num_samples_processed)
            self.dashboard.add_scalar("Train/NodeLoss", node_loss.item(), num_samples_processed)

            if num_batches_processed % self.validation_freq == 0:
                self.embedder.eval()
                if self.criterion.learnable_link_ranker:
                    self.link_ranker.eval()
                val_metrics, _ = self.compute_evaluation_metrics(self.val_set)
                for metric_name, metric_value in val_metrics.items():
                    self.dashboard.add_scalar(f"Validation/{metric_name}", metric_value, num_samples_processed)
                val_com_ap, val_ap = val_metrics["ComAP"], val_metrics["AP"]
                val_loss = 1 - 0.5 * (val_com_ap + val_ap)
                train_loss_change = abs(loss.item() - prev_train_loss)
                if (best_val_loss is not None and val_loss >= best_val_loss) and train_loss_change < self.val_tolerance:
                    patience += 1
                else:
                    patience = 0
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_embedder = deepcopy(self.embedder.state_dict())
                        best_link_ranker = deepcopy(self.link_ranker.state_dict())
                prev_train_loss = loss.item()
                self.dashboard.add_scalar("Validation/Patience", patience, num_samples_processed)
                if num_batches_processed % self.checkpoint_freq == 0:
                    checkpoint = {"start_time": start_time,
                                  "prev_train_loss": prev_train_loss,
                                  "best_val_loss": best_val_loss,
                                  "num_samples_processed": num_samples_processed,
                                  "num_batches_processed": num_batches_processed,
                                  "patience": patience,
                                  "query_index": self.loader.sampler.query_index,
                                  "answer_indices": self.loader.sampler.answer_indices,
                                  "answer_indices_c": self.loader.sampler.answer_indices_c,
                                  "embedder_state_dict": self.embedder.state_dict(),
                                  "link_ranker_state_dict": self.link_ranker.state_dict(),
                                  "embedder_optim_state_dict": self.embedder_optim.state_dict()}
                    if self.criterion.learnable_link_ranker:
                        checkpoint.update(link_ranker_optim_state_dict=self.link_ranker_optim.state_dict())
                    pt.save(checkpoint, f"graph_completion/results/{self.loader.dataset_name}/runs/"
                                        f"{self.run_id}/checkpoints/checkpoint_{num_batches_processed}.tar")
                self.dashboard.add_scalar("Train/Time", time.time() - start_time, num_samples_processed)

                with open(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}/train_log.txt",
                          mode="a+", encoding="utf-8") as train_log:
                    hparam_dict = {
                        "Seed": self.seed, "MiniBatchSize": self.mini_batch_size, "LearningRate": self.lr,
                        "Algorithm": self.algorithm, "CommunityMethod": self.community_method,
                        "LeidenResolution": self.leiden_resolution, "EmbeddingDim": self.embedder.embedding_dim,
                        "LossMargin": self.criterion.embedding_loss_hpars.margin,
                        "Dataset": self.loader.dataset_name, "NumNodes": self.loader.num_nodes,
                        "NumNodeTypes": self.loader.num_node_types,
                        "NumRelations": self.loader.num_relations,
                        "NumCommunities": self.loader.num_communities,
                        "NumNegativeSamples": self.loader.sampler.num_negative_samples
                    }
                    hparam_dict.update(**self.loader.graph_analysis_metrics)
                    train_line = "\t".join([str(v) for _, v in hparam_dict.items()])
                    train_line += f"\t{com_loss.item()}\t{node_loss.item()}\t{loss.item()}"
                    train_line += f"\t{time.time() - start_time}\t"
                    train_line += "\t".join([str(v) for _, v in val_metrics.items()])
                    train_line += f"\t{patience}\n"
                    train_log.write(train_line)

            for _, train_batch in train_batch_dict.items():
                num_samples_processed += len(train_batch)
            num_batches_processed += 1
            del train_batch_dict
            pt.cuda.empty_cache()

        if self.train:
            checkpoint = {"embedder_state_dict": best_embedder,
                          "link_ranker_state_dict": best_link_ranker}
            pt.save(checkpoint, f"graph_completion/results/{self.loader.dataset_name}/runs/"
                                f"{self.run_id}/checkpoints/checkpoint_best.tar")
        else:
            best_checkpoint = pt.load(f"graph_completion/results/{self.loader.dataset_name}/runs/"
                                      f"{self.checkpoint_run}/checkpoints/checkpoint_{self.checkpoint_tag}.tar",
                                      map_location=self.device)
            best_embedder = best_checkpoint["embedder_state_dict"]
            best_link_ranker = best_checkpoint["link_ranker_state_dict"]
        self.embedder.load_state_dict(best_embedder)
        self.link_ranker.load_state_dict(best_link_ranker)

    def compute_evaluation_metrics(self, evaluation_queries_dict: Dict[str, QueryData],
                                   query_answering: bool = False) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
        metrics = {}
        ranks = None

        # Link prediction metrics
        with pt.no_grad():
            for query_structure, evaluation_queries in tqdm(evaluation_queries_dict.items(),
                                                            "Query structure", leave=False):
                num_batches = ceil((evaluation_queries.query_tree_batch["sample"].max().item() + 1)
                                   / self.mini_batch_size)
                loss, com_loss, node_loss = [], [], []
                for evaluation_batch in tqdm(evaluation_queries.sample_split(self.mini_batch_size),
                                             "Computing loss", total=num_batches, leave=False):
                    loss_batch, (com_loss_batch, node_loss_batch) = self.criterion(
                        self.embedder.embed_supervised(evaluation_batch), self.link_ranker
                    )
                    loss.append(loss_batch * len(evaluation_batch) / len(evaluation_queries))
                    com_loss.append(com_loss_batch * len(evaluation_batch) / len(evaluation_queries))
                    node_loss.append(node_loss_batch * len(evaluation_batch) / len(evaluation_queries))
                metrics["ComLoss"] = pt.stack(com_loss).sum().item()
                metrics["NodeLoss"] = pt.stack(node_loss).sum().item()
                metrics["Loss"] = pt.stack(loss).sum().item()

                y_pred_c = []
                y_pred = []
                num_batches = ceil(len(evaluation_queries) / self.mini_batch_size)
                for evaluation_batch in tqdm(evaluation_queries.batch_split(self.mini_batch_size),
                                             "Classifying", total=num_batches, leave=False):
                    y_pred_c.append(self.link_ranker(*self.embedder.embed_communities(evaluation_batch),
                                                     for_communities=True))
                    y_pred.append(self.link_ranker(*self.embedder(evaluation_batch)))
                y_pred_c, y_pred = pt.cat(y_pred_c).cpu().numpy(), pt.cat(y_pred).cpu().numpy()
                y = evaluation_queries.query_tree_batch["y"].cpu().numpy()
                metrics["ComAccuracy"] = accuracy_score(y, y_pred_c > 0.5)
                metrics["Accuracy"] = accuracy_score(y, y_pred > 0.5)
                metrics["ComPrecision"] = precision_score(y, y_pred_c > 0.5, average="weighted", zero_division=0)
                metrics["Precision"] = precision_score(y, y_pred > 0.5, average="weighted", zero_division=0)
                metrics["ComRecall"] = recall_score(y, y_pred_c > 0.5, average="weighted", zero_division=0)
                metrics["Recall"] = recall_score(y, y_pred > 0.5, average="weighted", zero_division=0)
                metrics["ComF1"] = f1_score(y, y_pred_c > 0.5, average="weighted", zero_division=0)
                metrics["F1"] = f1_score(y, y_pred > 0.5, average="weighted", zero_division=0)
                metrics["ComROC-AUC"] = roc_auc_score(y, y_pred_c, average="weighted")
                metrics["ROC-AUC"] = roc_auc_score(y, y_pred, average="weighted")
                metrics["ComAP"] = average_precision_score(y, y_pred_c, average="weighted")
                metrics["AP"] = average_precision_score(y, y_pred, average="weighted")

                if query_answering:
                    # Query answering metrics
                    _, adj_s_to_t, adj_t_to_s, adj_s_to_t_c, adj_t_to_s_c = get_efficient_indexes(
                        self.loader.dataset.node_data, self.loader.dataset.edge_data, self.loader.communities
                    )
                    ranks = []
                    all_communities = pt.arange(self.loader.num_communities, device=self.device)
                    for pos_query in tqdm(evaluation_queries.get_only_positive(), "Query answering",
                                          total=pt.sum(evaluation_queries.query_tree_batch["y"]).item(), leave=False):
                        x_q = [x[0] for x in pos_query.query_tree_batch.vs["x"]]
                        e_q = [e[0] for e in pos_query.query_tree_batch.vs["e"]]
                        c_q = [c[0] for c in pos_query.query_tree_batch.vs["c"]]
                        r_q = [edge_attr[0] for edge_attr in pos_query.query_tree_batch.es["edge_attr"]]
                        c_answer = c_q[pos_query.query.query_answer]
                        anchors = pt.stack(e_q)[pos_query.query.query_anchors]
                        answer = e_q[pos_query.query.query_answer]
                        e_cpu = [e.item() for e in e_q]
                        c_cpu = [c.item() for c in c_q]
                        r_cpu = [f"p{edge_attr.argmax().item()}" if r == "p" else r
                                 for r, edge_attr in zip(pos_query.query_tree_batch.es["r"], r_q)]
                        qi_tree_c = pos_query.query.query_tree.copy()
                        qi_tree_c.vs["e"] = c_cpu
                        qi_tree_c.es["r"] = r_cpu
                        qi_tree = pos_query.query.query_tree.copy()
                        qi_tree.vs["e"] = e_cpu
                        qi_tree.es["r"] = r_cpu

                        # Community ranking
                        scores_c = []
                        all_answers_c = pt.tensor(
                            list(set(get_all_answers(qi_tree_c, pos_query.query, adj_s_to_t_c))),
                            dtype=pt.long, device=self.device
                        )
                        filtered_communities = all_communities[(all_communities == c_answer)
                                                               | (~pt.isin(all_communities, all_answers_c))]
                        e_c = [all_communities if i == pos_query.query.query_answer
                               else c_q[i].expand(self.loader.num_communities)
                               for i in range(pos_query.query_size)]
                        edge_index_c_q = pt.tensor(qi_tree_c.get_edge_dataframe()[["source", "target"]].values,
                                                   dtype=pt.long, device=self.device)
                        e_c_stacked = pt.stack(e_c)
                        edge_index_c_q = [e_c_stacked[edge_index_c] for edge_index_c in edge_index_c_q]
                        edge_attr_c = [edge_attr.expand(self.loader.num_communities, self.loader.num_relations + 1)
                                       for edge_attr in r_q]
                        n_c = [pt.tensor(self.loader.com_neighbours[:, query_edge_r_to_int(r)][
                                             edge_index_c.detach().cpu().numpy()
                                         ], dtype=pt.long, device=self.device)
                               for r, edge_index_c in zip(r_cpu, edge_index_c_q)]
                        all_queries = QueryData(query=pos_query.query, e_c=e_c, edge_attr_c=edge_attr_c, n_c=n_c)

                        num_batches = ceil(len(all_queries) / self.mini_batch_size)
                        for evaluation_batch in tqdm(all_queries.batch_split(self.mini_batch_size),
                                                     "Scoring communities", total=num_batches, leave=False):
                            scores_c.append(self.link_ranker(*self.embedder.embed_communities(evaluation_batch),
                                                             for_communities=True))
                        scores_c = pt.cat(scores_c)
                        scores_c[filtered_communities] += 1
                        rank_c_all = pt.sum(scores_c.unsqueeze(0) <= scores_c.unsqueeze(1), dim=0)
                        rank_c = rank_c_all[c_answer].item()

                        # Node ranking
                        scores = []
                        nodes_in_community = (self.embedder.community_membership == c_answer).nonzero().squeeze(dim=1)
                        all_answers = pt.tensor(
                            list(set(get_all_answers(qi_tree, pos_query.query, adj_s_to_t))),
                            dtype=pt.long, device=self.device
                        )
                        filtered_nodes = nodes_in_community[(nodes_in_community == answer)
                                                            | ((~pt.isin(nodes_in_community, all_answers))
                                                               & (~pt.isin(nodes_in_community, anchors)))]
                        community_size_filtered = len(filtered_nodes)
                        x = [one_hot(self.embedder.node_types[filtered_nodes], self.loader.num_node_types).float()
                             if i == pos_query.query.query_answer else
                             x_q[i].expand(community_size_filtered, self.loader.num_node_types)
                             for i in range(pos_query.query_size)]
                        e = [filtered_nodes if i == pos_query.query.query_answer
                             else e_q[i].expand(community_size_filtered)
                             for i in range(pos_query.query_size)]
                        edge_index_q = pt.tensor(qi_tree.get_edge_dataframe()[["source", "target"]].values,
                                                 dtype=pt.long, device=self.device)
                        e_stacked = pt.stack(e)
                        edge_index_q = [e_stacked[edge_index] for edge_index in edge_index_q]
                        edge_attr = [edge_attr.expand(community_size_filtered, self.loader.num_relations + 1)
                                     for edge_attr in r_q]
                        n = [pt.tensor(self.loader.node_neighbours[:, query_edge_r_to_int(r)][
                                           edge_index.detach().cpu().numpy()
                                       ], dtype=pt.long, device=self.device)
                             for r, edge_index in zip(r_cpu, edge_index_q)]
                        for i in range(pos_query.query_size - 1):
                            query_tree_edge = qi_tree.es[i]
                            s_q, t_q = e_q[query_tree_edge.source], e_q[query_tree_edge.target]
                            c_s_q, c_t_q = c_q[query_tree_edge.source], c_q[query_tree_edge.target]
                            n[i][0, self.embedder.community_membership[n[i][0]] != c_t_q] = s_q
                            n[i][1, self.embedder.community_membership[n[i][1]] != c_s_q] = t_q
                        c = [c.expand(community_size_filtered) for c in c_q]
                        all_triplets = QueryData(query=pos_query.query, e=e, x=x, edge_attr=edge_attr, n=n, c=c)

                        num_batches = ceil(len(all_triplets) / self.mini_batch_size)
                        for evaluation_batch in tqdm(all_triplets.batch_split(self.mini_batch_size),
                                                     "Scoring nodes", total=num_batches, leave=False):
                            scores.append(self.link_ranker(*self.embedder(evaluation_batch)))
                        scores = pt.cat(scores)
                        intra_community_rank = pt.sum(scores[filtered_nodes == answer] <= scores).item()

                        c_err = self.loader.community_sizes[(rank_c_all < rank_c).cpu().numpy()].sum()
                        rank = c_err + intra_community_rank
                        ranks.append((self.loader.num_communities, len(filtered_communities),
                                      len(nodes_in_community), community_size_filtered,
                                      rank_c, intra_community_rank, rank, query_structure))

                    ranks = pd.DataFrame(ranks, columns=["NumCommunities", "FilteredCommunities",
                                                         "NumNodes", "FilteredNodes",
                                                         "ComRank", "NodeRank", "Rank", "Query"])
                    metrics["ComHits@1"] = np.mean(ranks.ComRank.values <= 1)
                    metrics["NodeHits@1"] = np.mean(ranks.NodeRank.values <= 1)
                    metrics["Hits@1"] = np.mean(ranks.Rank.values <= 1)
                    metrics["ComHits@3"] = np.mean(ranks.ComRank.values <= 3)
                    metrics["NodeHits@3"] = np.mean(ranks.NodeRank.values <= 3)
                    metrics["Hits@3"] = np.mean(ranks.Rank.values <= 3)
                    metrics["ComHits@10"] = np.mean(ranks.ComRank.values <= 10)
                    metrics["NodeHits@10"] = np.mean(ranks.NodeRank.values <= 10)
                    metrics["Hits@10"] = np.mean(ranks.Rank.values <= 10)
                    metrics["ComMR"] = np.mean(ranks.ComRank.values)
                    metrics["NodeMR"] = np.mean(ranks.NodeRank.values)
                    metrics["MR"] = np.mean(ranks.Rank.values)
                    metrics["ComMRR"] = np.mean(1 / ranks.ComRank.values)
                    metrics["NodeMRR"] = np.mean(1 / ranks.NodeRank.values)
                    metrics["MRR"] = np.mean(1 / ranks.Rank.values)

        return metrics, ranks

    def main(self):
        self.prepare()

        self.run_training()

        print("Testing...")
        self.embedder.eval()
        if self.criterion.learnable_link_ranker:
            self.link_ranker.eval()
        for checkpoint_file in glob(f"graph_completion/results/{self.loader.dataset_name}/"
                                    f"runs/{self.run_id if self.train else self.checkpoint_run}/"
                                    f"checkpoints/checkpoint_*.tar"):
            checkpoint_tag = checkpoint_file[:-4].split("/")[-1].split("_")[-1]
            checkpoint = pt.load(checkpoint_file, map_location=self.device)
            embedder = checkpoint["embedder_state_dict"]
            link_ranker = checkpoint["link_ranker_state_dict"]
            self.embedder.load_state_dict(embedder)
            self.link_ranker.load_state_dict(link_ranker)
            start_time = time.time()
            test_metrics, test_ranks = self.compute_evaluation_metrics(self.test_set, query_answering=True)
            test_metrics.update(Time=time.time() - start_time)
            test_ranks = test_ranks.assign(Dataset=self.loader.dataset_name, Algorithm=self.algorithm,
                                           Seed=self.seed, Checkpoint=checkpoint_tag)
            if checkpoint_tag == "best":
                for metric_name, metric_value in test_metrics.items():
                    self.dashboard.add_scalar(f"Test/{metric_name}", metric_value, self.loader.num_nodes)
            hparam_dict = {
                "Seed": self.seed, "MiniBatchSize": self.mini_batch_size, "LearningRate": self.lr,
                "Algorithm": self.algorithm, "CommunityMethod": self.community_method,
                "LeidenResolution": self.leiden_resolution, "EmbeddingDim": self.embedder.embedding_dim,
                "LossMargin": self.criterion.embedding_loss_hpars.margin,
                "Dataset": self.loader.dataset_name, "NumNodes": self.loader.num_nodes,
                "NumNodeTypes": self.loader.num_node_types,
                "NumRelations": self.loader.num_relations,
                "NumCommunities": self.loader.num_communities,
                "NumNegativeSamples": self.loader.sampler.num_negative_samples
            }
            hparam_dict.update(**self.loader.graph_analysis_metrics)
            if checkpoint_tag == "best":
                self.dashboard.add_hparams(hparam_dict=hparam_dict, metric_dict=test_metrics,
                                           name=f"{dirname(realpath(__file__))}/results/"
                                                f"{self.loader.dataset_name}/runs/{self.run_id}")
                if self.algorithm == "transe":
                    pt.save(self.embedder.state_dict(),
                            f"graph_completion/results/{self.loader.dataset_name}/transe_model.tar")
            with open(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}/test_log.txt",
                      mode="a+", encoding="utf-8") as test_log:
                test_line = f"{checkpoint_tag}\t"
                test_line += "\t".join([str(v) for _, v in hparam_dict.items()])
                test_line += "\t" + "\t".join([str(v) for _, v in test_metrics.items()]) + "\n"
                test_log.write(test_line)
            test_ranks.to_csv(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}/test_ranks.txt",
                              sep="\t", mode="a+", encoding="utf-8", header=False, index=False)


@attr.s
class ExperimentHpars(AbstractConf):
    OPTIONS = {"experiment": Experiment}
    seed = attr.ib(default=123456789, validator=instance_of(int))
    device = attr.ib(default="cpu", validator=in_(["cpu", ] + [f"cuda:{i}" for i in range(device_count())]))
    train = attr.ib(default=True, validator=instance_of(bool))
    checkpoint_run = attr.ib(default=0, validator=instance_of(int))
    checkpoint_tag = attr.ib(default="", validator=instance_of(str))
    val_size = attr.ib(default=0.01, validator=and_(instance_of(float), ge(0), le(1)))
    test_size = attr.ib(default=0.02, validator=and_(instance_of(float), ge(0), le(1)))
    mini_batch_size = attr.ib(default=25, validator=instance_of(int))
    lr = attr.ib(default=1e-3, validator=instance_of(float))
    weight_decay = attr.ib(default=1e-6, validator=instance_of(float))
    val_patience = attr.ib(default=50, validator=instance_of(int))
    val_tolerance = attr.ib(default=1e-4, validator=instance_of(float))
    max_epochs = attr.ib(default=5, validator=instance_of(int))
    validation_freq = attr.ib(default=1, validator=instance_of(int))
    checkpoint_freq = attr.ib(default=1000, validator=instance_of(int))
    algorithm = attr.ib(default="transe", validator=in_(["mlp", "transe", "distmult", "complex", "rotate",
                                                         "gatne", "sacn", "kbgat", "gqe", "q2b", "betae"]))
    community_method = attr.ib(default="leiden", validator=in_(["random", "metis", "leiden"]))
    leiden_resolution = attr.ib(default=0.0, validator=instance_of(float))
    coins_alpha = attr.ib(default=0.5, validator=instance_of(float))
    transe_initialize = attr.ib(default=False, validator=instance_of(bool))
    loader_hpars = attr.ib(factory=LoaderHpars, validator=lambda i, a, v: type(v) is LoaderHpars)
    embedder_hpars = attr.ib(factory=GraphEmbedderHpars, validator=lambda i, a, v: type(v) is GraphEmbedderHpars)
    link_ranker_hpars = attr.ib(factory=LinkRankerHpars, validator=lambda i, a, v: type(v) is LinkRankerHpars)
    embedding_loss_hpars = attr.ib(factory=EmbeddingLossHpars, validator=lambda i, a, v: type(v) is EmbeddingLossHpars)
    name = "experiment"

    def __attrs_post_init__(self):
        self.embedder_hpars["algorithm"] = self.algorithm
        self.link_ranker_hpars["algorithm"] = self.algorithm
        self.embedding_loss_hpars["algorithm"] = self.algorithm
