import pandas as pd
from ogb.linkproppred import LinkPropPredDataset
from ogb.lsc import WikiKG90Mv2Dataset


def download_preprocess_store_ogbl_citation2():
    dataset = LinkPropPredDataset("ogbl-citation2", "data/ogb/ogbl-citation2")

    train_test_split = dataset.get_edge_split()
    train_edge_data = pd.DataFrame().assign(s=train_test_split["train"]["source_node"], r="cites",
                                            t=train_test_split["train"]["target_node"])
    train_edge_data.to_csv("data/ogb/ogbl-citation2/train_edges.csv", sep=",",
                           header=True, index=False, mode="w", encoding="utf-8")
    val_edge_data = pd.DataFrame().assign(s=train_test_split["valid"]["source_node"], r="cites",
                                          t=train_test_split["valid"]["target_node"])
    val_edge_data.to_csv("data/ogb/ogbl-citation2/val_edges.csv", sep=",",
                         header=True, index=False, mode="w", encoding="utf-8")
    test_edge_data = pd.DataFrame().assign(s=train_test_split["test"]["source_node"], r="cites",
                                           t=train_test_split["test"]["target_node"])
    test_edge_data.to_csv("data/ogb/ogbl-citation2/test_edges.csv", sep=",",
                          header=True, index=False, mode="w", encoding="utf-8")
    dataset = dataset[0]
    del train_test_split, train_edge_data, val_edge_data, test_edge_data

    edge_data = pd.DataFrame(dataset["edge_index"].T, columns=["s", "t"]).assign(r="cites")

    node_data = pd.concat((edge_data.s, edge_data.t)).to_frame(name="n").drop_duplicates(ignore_index=True)
    node_time = pd.to_datetime(pd.Series(dataset["node_year"][:, 0]), format="%Y")
    node_data = node_data.assign(time=node_data.n.map(node_time), type="paper")

    edge_data = edge_data.merge(node_data, how="left", left_on="s", right_on="n")[["s", "r", "t", "time"]]

    node_data.to_csv("data/ogb/ogbl-citation2/nodes.csv", sep=",", header=True, index=False, mode="w", encoding="utf-8")
    edge_data.to_csv("data/ogb/ogbl-citation2/edges.csv", sep=",", header=True, index=False, mode="w", encoding="utf-8")


def download_preprocess_store_ogbl_biokg():
    dataset = LinkPropPredDataset("ogbl-biokg", "data/ogb/ogbl-biokg")
    train_test_split = dataset.get_edge_split()
    r_map = pd.Series({data[0, 0]: r for (_, r, _), data in dataset[0]["edge_reltype"].items()}, name="r")

    train_edge_data = pd.DataFrame().assign(s_type=train_test_split["train"]["head_type"],
                                            s=train_test_split["train"]["head"],
                                            r=r_map[train_test_split["train"]["relation"]].values,
                                            t=train_test_split["train"]["tail"],
                                            t_type=train_test_split["train"]["tail_type"])
    val_edge_data = pd.DataFrame().assign(s_type=train_test_split["valid"]["head_type"],
                                          s=train_test_split["valid"]["head"],
                                          r=r_map[train_test_split["valid"]["relation"]].values,
                                          t=train_test_split["valid"]["tail"],
                                          t_type=train_test_split["valid"]["tail_type"], )
    test_edge_data = pd.DataFrame().assign(s_type=train_test_split["test"]["head_type"],
                                           s=train_test_split["test"]["head"],
                                           r=r_map[train_test_split["test"]["relation"]].values,
                                           t=train_test_split["test"]["tail"],
                                           t_type=train_test_split["test"]["tail_type"], )
    del train_test_split

    edge_data = pd.concat((train_edge_data, val_edge_data, test_edge_data), ignore_index=True)
    node_data = pd.concat((
        edge_data[["s", "s_type"]].rename(columns={"s": "n", "s_type": "type"}),
        edge_data[["t", "t_type"]].rename(columns={"t": "n", "t_type": "type"}))
    ).drop_duplicates(ignore_index=True)
    node_id_fix = node_data.type.value_counts().cumsum() - node_data.type.value_counts()
    node_data = node_data.assign(n=node_data.type.map(node_id_fix).rename("n") + node_data.n, time=pd.NaT)
    node_data.to_csv("data/ogb/ogbl-biokg/nodes.csv", sep=",", header=True, index=False, mode="w", encoding="utf-8")

    for file_name, edges in {"train_edges": train_edge_data, "val_edges": val_edge_data,
                             "test_edges": test_edge_data, "edges": edge_data}.items():
        edges = edges.assign(s=edges.s_type.map(node_id_fix.rename("s_type")).rename("s") + edges.s,
                             t=edges.t_type.map(node_id_fix.rename("t_type")).rename("t") + edges.t)
        edges = edges.drop(columns=["s_type", "t_type"]).assign(time=pd.NaT)
        edges.to_csv(f"data/ogb/ogbl-biokg/{file_name}.csv", sep=",",
                     header=True, index=False, mode="w", encoding="utf-8")


def download_preprocess_store_ogb_lsc_wikikg90m2():
    dataset = WikiKG90Mv2Dataset("data/ogb/ogb-lsc-wikikg90m2")
    train_edge_data = pd.DataFrame(dataset.train_hrt, columns=["s", "r", "t"]).assign(time=pd.NaT)
    val_edge_data = pd.DataFrame(dataset.valid_dict["h,r->t"]["hr"], columns=["s", "r"])
    val_edge_data = val_edge_data.assign(t=dataset.valid_dict["h,r->t"]["t"], time=pd.NaT)

    edge_data = pd.concat((train_edge_data, val_edge_data), ignore_index=True)
    node_data = pd.DataFrame(pd.concat(
        (edge_data.s.rename("n"), edge_data.t.rename("n"))
    ).unique(), columns=["n", ]).assign(type="entity", time=pd.NaT)
    node_data.to_csv("data/ogb/ogb-lsc-wikikg90m2/nodes.csv", sep=",",
                     header=True, index=False, mode="w", encoding="utf-8")
    edge_data.to_csv("data/ogb/ogb-lsc-wikikg90m2/edges.csv", sep=",",
                     header=True, index=False, mode="w", encoding="utf-8")

    del node_data, edge_data

    val_nodes = pd.concat((val_edge_data.s.rename("n"), val_edge_data.t.rename("n"))).unique()
    train_edge_data = train_edge_data[train_edge_data.s.isin(val_nodes) & train_edge_data.t.isin(val_nodes)]

    train_edge_data.to_csv("data/ogb/ogb-lsc-wikikg90m2/train_edges.csv", sep=",",
                           header=True, index=False, mode="w", encoding="utf-8")
    val_edge_data.to_csv("data/ogb/ogb-lsc-wikikg90m2/val_edges.csv", sep=",",
                         header=True, index=False, mode="w", encoding="utf-8")
    val_edge_data.to_csv("data/ogb/ogb-lsc-wikikg90m2/test_edges.csv", sep=",",
                         header=True, index=False, mode="w", encoding="utf-8")
