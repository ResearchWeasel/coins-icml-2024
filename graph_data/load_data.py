from abc import ABC, abstractmethod
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu
from sklearn.model_selection import train_test_split


class Dataset(ABC):
    @abstractmethod
    def __init__(self, name: str, file_location: str, *args, **kwargs):
        self.name: str = name
        self.file_location: str = file_location
        self.edge_data: pd.DataFrame = pd.DataFrame()
        self.node_data: pd.DataFrame = pd.DataFrame()
        self.node_names_map: pd.Series = pd.Series(dtype="int64")
        self.node_types_map: pd.Series = pd.Series(dtype="int64")
        self.relation_names_map: pd.Series = pd.Series(dtype="int64")
        pass

    @abstractmethod
    def load_from_disk(self, *args, **kwargs):
        pass

    def unload_from_memory(self):
        self.node_data = pd.DataFrame()
        self.edge_data = pd.DataFrame()

    def time_sort_and_numerize(self):
        self.node_data = self.node_data.sort_values(["time", "n"], ignore_index=True)
        self.edge_data = self.edge_data.sort_values(["time", "s", "t"], ignore_index=True)

        if len(self.node_names_map) == 0:
            self.node_names_map = self.node_data.n.reset_index(name="n").set_index("n")["index"]
        if len(self.node_types_map) == 0:
            node_types = self.node_data.type.drop_duplicates().sort_values(ignore_index=True)
            self.node_types_map = node_types.reset_index(name="type").set_index("type")["index"]
        self.node_data.n = self.node_data.n.map(self.node_names_map)
        self.node_data.type = self.node_data.type.map(self.node_types_map)
        self.edge_data.s = self.edge_data.s.map(self.node_names_map.rename("s"))
        self.edge_data.t = self.edge_data.t.map(self.node_names_map.rename("t"))

        if len(self.relation_names_map) == 0:
            relation_names = self.edge_data.r.drop_duplicates().sort_values(ignore_index=True)
            self.relation_names_map = relation_names.reset_index(name="r").set_index("r")["index"]
        self.edge_data.r = self.edge_data.r.map(self.relation_names_map)

    def get_graph_data_batch_generator(self, time_interval: pd.Timedelta) -> Generator[
        Tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame], None, None
    ]:
        in_time_interval = lambda df, st, et: (st <= df.time) & (df.time < et)
        node_start_time = self.node_data.time.iloc[0]
        edge_start_time = self.edge_data.time.iloc[0] if len(self.edge_data) > 0 else pd.NaT
        start_time = min(node_start_time, edge_start_time)
        if pd.isna(start_time):
            yield pd.to_datetime(0), self.node_data.drop(columns="time"), self.edge_data.drop(columns="time")
        else:
            num_nodes_returned = num_edges_returned = 0
            while num_nodes_returned != len(self.node_data) and num_edges_returned != len(self.edge_data):
                node_batch = self.node_data[in_time_interval(self.node_data, start_time, start_time + time_interval)]
                edge_batch = self.edge_data[in_time_interval(self.edge_data, start_time, start_time + time_interval)]
                yield start_time, node_batch, edge_batch
                num_nodes_returned += len(node_batch)
                num_edges_returned += len(edge_batch)
                start_time += time_interval

    def get_community_generator(self, community_membership: np.ndarray, num_communities: int) -> Generator[
        Tuple[int, pd.DataFrame, pd.DataFrame], None, None
    ]:
        for community_id in range(num_communities):
            community_node_idx = np.nonzero(community_membership == community_id)[0]
            node_data = self.node_data.iloc[community_node_idx].reset_index(drop=True)
            edge_data = self.edge_data[self.edge_data.s.isin(node_data.n)
                                       & self.edge_data.t.isin(node_data.n)].reset_index(drop=True)
            if len(edge_data) > 0:
                node_names_map = self.node_names_map.iloc[community_node_idx]
                intra_community_node_id_map = node_data.n.reset_index(name="n").set_index("n")["index"]
                node_names_map = node_names_map.map(intra_community_node_id_map)
                node_data.n = node_data.n.map(intra_community_node_id_map)
                edge_data.s = edge_data.s.map(intra_community_node_id_map.rename("s"))
                edge_data.t = edge_data.t.map(intra_community_node_id_map.rename("t"))
                yield community_id, node_names_map, node_data, edge_data

    def get_overlapping_region_generator(self, community_membership: csr_matrix) -> Generator[
        Tuple[int, int, pd.DataFrame, pd.DataFrame], None, None
    ]:
        region_sizes = triu(community_membership.T @ community_membership, k=1)
        for community_id, community_id_2 in zip(*(region_sizes >= 10).nonzero()):
            region_node_idx = np.nonzero(
                community_membership.getcol(community_id).multiply(community_membership.getcol(community_id_2))
            )[0]
            node_data = self.node_data.iloc[region_node_idx].reset_index(drop=True)
            edge_data = self.edge_data[self.edge_data.s.isin(node_data.n)
                                       & self.edge_data.t.isin(node_data.n)].reset_index(drop=True)
            if len(edge_data) > 0:
                node_names_map = self.node_names_map.iloc[region_node_idx]
                intra_region_node_id_map = node_data.n.reset_index(name="n").set_index("n")["index"]
                node_names_map = node_names_map.map(intra_region_node_id_map)
                node_data.n = node_data.n.map(intra_region_node_id_map)
                edge_data.s = edge_data.s.map(intra_region_node_id_map.rename("s"))
                edge_data.t = edge_data.t.map(intra_region_node_id_map.rename("t"))
                yield community_id, community_id_2, node_names_map, node_data, edge_data

    def clean_split(self, edge_data: pd.DataFrame) -> pd.DataFrame:
        edge_data = edge_data.sort_values(["time", "s", "t"], ignore_index=True)
        return edge_data

    def train_val_test_split(self, val_size: float, test_size: float,
                             seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_size = min(int(test_size * len(self.edge_data)), 10000)
        val_size = min(int(val_size * len(self.edge_data)), 5000)
        labels = self.edge_data.r
        label_freqs = labels.value_counts()
        unsplittable = labels.isin(label_freqs[label_freqs < 2].index)
        train_val_edge_data, test_edge_data = train_test_split(self.edge_data[~unsplittable], test_size=test_size,
                                                               random_state=seed, shuffle=True,
                                                               stratify=labels[~unsplittable])
        labels = train_val_edge_data.r
        label_freqs = labels.value_counts()
        unsplittable = labels.isin(label_freqs[label_freqs < 2].index)
        train_edge_data, val_edge_data = train_test_split(train_val_edge_data[~unsplittable], test_size=val_size,
                                                          random_state=seed, shuffle=True,
                                                          stratify=labels[~unsplittable])
        del train_val_edge_data
        return self.clean_split(train_edge_data), self.clean_split(val_edge_data), self.clean_split(test_edge_data)


class OGBBioKG(Dataset):
    def __init__(self):
        super().__init__("ogbl-biokg", f"data/ogb/ogbl-biokg")

    def load_from_disk(self):
        self.node_data = pd.read_csv(f"{self.file_location}/nodes.csv",
                                     sep=",", header=0, index_col=None, encoding="utf-8", parse_dates=["time", ])
        self.edge_data = pd.read_csv(f"{self.file_location}/edges.csv",
                                     sep=",", header=0, index_col=None, encoding="utf-8", parse_dates=["time", ])

    def clean_split(self, edge_data: pd.DataFrame) -> pd.DataFrame:
        edge_data = edge_data.assign(time=pd.NaT)
        edge_data.s = edge_data.s.map(self.node_names_map.rename("s"))
        edge_data.t = edge_data.t.map(self.node_names_map.rename("t"))
        edge_data.r = edge_data.r.map(self.relation_names_map)

        return super().clean_split(edge_data)

    def train_val_test_split(self, val_size: float, test_size: float,
                             seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_edges = pd.read_csv(f"{self.file_location}/train_edges.csv",
                                  sep=",", header=0, index_col=None, encoding="utf-8")
        valid_edges = pd.read_csv(f"{self.file_location}/val_edges.csv",
                                  sep=",", header=0, index_col=None, encoding="utf-8")
        test_edges = pd.read_csv(f"{self.file_location}/test_edges.csv",
                                 sep=",", header=0, index_col=None, encoding="utf-8")
        return self.clean_split(train_edges), self.clean_split(valid_edges), self.clean_split(test_edges)


class OGBLSCWikiKG90Mv2(Dataset):
    def __init__(self):
        super().__init__("ogb-lsc-wikikg90m2", f"data/ogb/ogb-lsc-wikikg90m2")

    def load_from_disk(self):
        self.node_data = pd.read_csv(f"{self.file_location}/nodes.csv",
                                     sep=",", header=0, index_col=None, encoding="utf-8", parse_dates=["time", ])
        self.edge_data = pd.read_csv(f"{self.file_location}/edges.csv",
                                     sep=",", header=0, index_col=None, encoding="utf-8", parse_dates=["time", ])

    def clean_split(self, edge_data: pd.DataFrame) -> pd.DataFrame:
        edge_data = edge_data.assign(time=pd.NaT)
        edge_data.s = edge_data.s.map(self.node_names_map.rename("s"))
        edge_data.t = edge_data.t.map(self.node_names_map.rename("t"))
        edge_data.r = edge_data.r.map(self.relation_names_map)

        return super().clean_split(edge_data)

    def train_val_test_split(self, val_size: float, test_size: float,
                             seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_edges = pd.read_csv(f"{self.file_location}/train_edges.csv",
                                  sep=",", header=0, index_col=None, encoding="utf-8")
        valid_edges = pd.read_csv(f"{self.file_location}/val_edges.csv",
                                  sep=",", header=0, index_col=None, encoding="utf-8")
        test_edges = pd.read_csv(f"{self.file_location}/test_edges.csv",
                                 sep=",", header=0, index_col=None, encoding="utf-8")
        return self.clean_split(train_edges), self.clean_split(valid_edges), self.clean_split(test_edges)


class Transport(Dataset):
    def __init__(self, city="rome"):
        super().__init__(f"transport-{city}", f"data/transport/{city}")
        self.city = city

    def load_from_disk(self):
        node_data = pd.read_csv(f"{self.file_location}/network_nodes.csv",
                                sep=";", header=0, index_col=None, encoding="utf-8",
                                usecols=["stop_I", ])
        edge_data = pd.read_csv(f"{self.file_location}/network_combined.csv",
                                sep=";", header=0, index_col=None, encoding="utf-8",
                                usecols=["from_stop_I", "to_stop_I", "route_type"])

        self.node_data = node_data.rename(columns={"stop_I": "n"}).assign(type="stop", time=pd.NaT)
        self.edge_data = edge_data.rename(columns={"from_stop_I": "s",
                                                   "to_stop_I": "t",
                                                   "route_type": "r"}).assign(time=pd.NaT)
        self.edge_data = self.edge_data.assign(r=self.edge_data.r.map({
            0: "tram", 1: "subway", 2: "rail", 3: "bus", 4: "ferry", 5: "cablecar", 6: "gondola", 7: "funicular"
        }))


class FreeBase(Dataset):
    def __init__(self):
        super().__init__("fb15k-237", "data/academic/FB15k-237")

    def load_from_disk(self, *args, **kwargs):
        train_edges = pd.read_csv(f"{self.file_location}/train.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        valid_edges = pd.read_csv(f"{self.file_location}/valid.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        test_edges = pd.read_csv(f"{self.file_location}/test.txt",
                                 sep="\t", header=None, index_col=None, encoding="utf-8")
        edge_data = pd.concat((train_edges, valid_edges, test_edges), ignore_index=True)
        del train_edges, valid_edges, test_edges
        edge_data.columns = ["s", "r", "t"]
        self.edge_data = edge_data.assign(time=pd.NaT)
        node_data = pd.concat((edge_data.s, edge_data.t), ignore_index=True).to_frame(name="n")
        node_data = node_data.drop_duplicates(ignore_index=True)
        self.node_data = node_data.assign(type=node_data.n.str.split("/").map(lambda x: x[1]), time=pd.NaT)

    def clean_split(self, edge_data: pd.DataFrame) -> pd.DataFrame:
        edge_data.columns = ["s", "r", "t"]
        edge_data = edge_data.assign(time=pd.NaT)
        edge_data.s = edge_data.s.map(self.node_names_map.rename("s"))
        edge_data.t = edge_data.t.map(self.node_names_map.rename("t"))
        edge_data.r = edge_data.r.map(self.relation_names_map)

        return super().clean_split(edge_data)

    def train_val_test_split(self, val_size: float, test_size: float,
                             seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_edges = pd.read_csv(f"{self.file_location}/train.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        valid_edges = pd.read_csv(f"{self.file_location}/valid.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        test_edges = pd.read_csv(f"{self.file_location}/test.txt",
                                 sep="\t", header=None, index_col=None, encoding="utf-8")
        return self.clean_split(train_edges), self.clean_split(valid_edges), self.clean_split(test_edges)


class WordNet(Dataset):
    def __init__(self):
        super().__init__("wn18rr", "data/academic/WN18RR")

    def load_from_disk(self, *args, **kwargs):
        train_edges = pd.read_csv(f"{self.file_location}/train.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        valid_edges = pd.read_csv(f"{self.file_location}/valid.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        test_edges = pd.read_csv(f"{self.file_location}/test.txt",
                                 sep="\t", header=None, index_col=None, encoding="utf-8")
        edge_data = pd.concat((train_edges, valid_edges, test_edges), ignore_index=True)
        del train_edges, valid_edges, test_edges
        edge_data.columns = ["s", "r", "t"]
        self.edge_data = edge_data.assign(time=pd.NaT)
        node_data = pd.concat((edge_data.s, edge_data.t), ignore_index=True).to_frame(name="n")
        node_data = node_data.drop_duplicates(ignore_index=True)
        self.node_data = node_data.assign(type=node_data.n.str.split(".").map(lambda x: x[-2]), time=pd.NaT)

    def clean_split(self, edge_data: pd.DataFrame) -> pd.DataFrame:
        edge_data.columns = ["s", "r", "t"]
        edge_data = edge_data.assign(time=pd.NaT)
        edge_data.s = edge_data.s.map(self.node_names_map.rename("s"))
        edge_data.t = edge_data.t.map(self.node_names_map.rename("t"))
        edge_data.r = edge_data.r.map(self.relation_names_map)

        return super().clean_split(edge_data)

    def train_val_test_split(self, val_size: float, test_size: float,
                             seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_edges = pd.read_csv(f"{self.file_location}/train.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        valid_edges = pd.read_csv(f"{self.file_location}/valid.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        test_edges = pd.read_csv(f"{self.file_location}/test.txt",
                                 sep="\t", header=None, index_col=None, encoding="utf-8")
        return self.clean_split(train_edges), self.clean_split(valid_edges), self.clean_split(test_edges)


class NELL(Dataset):
    def __init__(self):
        super().__init__("nell-995", "data/academic/NELL-995")

    def load_from_disk(self, *args, **kwargs):
        edge_data = pd.read_csv(f"{self.file_location}/raw.kb",
                                sep="\t", header=None, index_col=None, encoding="utf-8")
        edge_data.columns = ["s", "r", "t"]
        self.edge_data = edge_data.assign(time=pd.NaT)
        node_data = pd.concat((edge_data.s, edge_data.t), ignore_index=True).to_frame(name="n")
        node_data = node_data.drop_duplicates(ignore_index=True)
        self.node_data = node_data.assign(type=node_data.n.map(lambda x: x.split(":")[1] if ":" in x else "numeric"),
                                          time=pd.NaT)

    def clean_split(self, edge_data: pd.DataFrame) -> pd.DataFrame:
        edge_data.columns = ["s", "r", "t"]
        edge_data = edge_data.assign(time=pd.NaT)
        edge_data.s = edge_data.s.str.replace("_", ":", n=2).map(self.node_names_map.rename("s"))
        edge_data.t = edge_data.t.str.replace("_", ":", n=2).map(self.node_names_map.rename("t"))
        edge_data.r = edge_data.r.map(self.relation_names_map)

        return super().clean_split(edge_data)

    def train_val_test_split(self, val_size: float, test_size: float,
                             seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_edges = pd.read_csv(f"{self.file_location}/train.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        valid_edges = pd.read_csv(f"{self.file_location}/valid.txt",
                                  sep="\t", header=None, index_col=None, encoding="utf-8")
        test_edges = pd.read_csv(f"{self.file_location}/test.txt",
                                 sep="\t", header=None, index_col=None, encoding="utf-8")
        return self.clean_split(train_edges), self.clean_split(valid_edges), self.clean_split(test_edges)
