from glob import glob
from math import sqrt
from os.path import exists
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from graph_analysis.metrics import Subgraphing
from graph_completion.experiments import ExperimentHpars
from graph_completion.graphs.load_graph import LoaderHpars
from graph_completion.utils import reproduce

sns.set(context="paper", font_scale=5, style="darkgrid")


def parse_baseline_metrics(dataset: str, algorithm: str) -> Tuple[float, float, float, float, float]:
    if algorithm == "KBGAT":
        return kbgat_baselines[dataset]
    with open(f"graph_completion/results_baseline/{algorithm}_{dataset}_0/train.log",
              mode="r", encoding="utf-8") as baseline_log_file:
        baseline_log = baseline_log_file.read().splitlines()
        baseline_hits_at_1 = float(baseline_log[-3].split(" ")[-1])
        baseline_hits_at_3 = float(baseline_log[-2].split(" ")[-1])
        baseline_hits_at_10 = float(baseline_log[-1].split(" ")[-1])
        baseline_mr = float(baseline_log[-4].split(" ")[-1])
        baseline_mrr = float(baseline_log[-5].split(" ")[-1])
    return baseline_hits_at_1, baseline_hits_at_3, baseline_hits_at_10, baseline_mr, baseline_mrr


def calculate_community_method_metrics(subgraphing: Subgraphing):
    communities = subgraphing.metric_values["DisjointCommunityMembership"][-1]
    num_communities = int(subgraphing.metric_values['DisjointCommunityNumber'][-1])
    community_sizes = np.rint(
        dataset_loader.num_nodes * subgraphing.metric_values["DisjointCommunitySizeDist"][-1][:num_communities]
    ).astype(int)
    c_s = communities[dataset_loader.test_edge_data.s]
    c_t = communities[dataset_loader.test_edge_data.t]

    test_edges_communities = c_t[np.argsort(num_communities * c_t + c_s)]
    com_test_edges = dataset_loader.test_edge_data.assign(c_t=c_t).groupby("c_t").size().reindex(
        np.arange(num_communities), fill_value=0
    ).values
    std_community_sizes, std_com_test_edges = np.std(community_sizes), np.std(com_test_edges)
    cut_size = subgraphing.metric_values['DisjointCommunityCutSize'][-1]
    modularity = subgraphing.metric_values['DisjointCommunityModularity'][-1]
    speed_up = ((N[dataset_key] * V_size[dataset_key])
                / ((num_communities + community_sizes) * com_test_edges).sum())
    inter_community_edges = (communities[dataset_loader.dataset.edge_data.s]
                             != communities[dataset_loader.dataset.edge_data.t])
    inter_community_nodes = np.unique(np.concatenate((
        dataset_loader.dataset.edge_data[inter_community_edges].s.values,
        dataset_loader.dataset.edge_data[inter_community_edges].t.values)
    ))
    bulk_up = (V_size[dataset_key] + num_communities + len(inter_community_nodes) + 1) / V_size[dataset_key]
    return ((std_community_sizes, std_com_test_edges, cut_size, modularity, speed_up, bulk_up),
            (test_edges_communities, com_test_edges))


if __name__ == "__main__":
    datasets_names = {"freebase": "FB15k-237", "wordnet": "WN18RR", "nell": "NELL-995"}
    algs_names = {"transe": "TransE", "distmult": "DistMult",
                  "complex": "ComplEx", "rotate": "RotatE", "kbgat": "KBGAT"}
    kbgat_baselines = {"freebase": (0.46, 0.54, 0.626, 210, 0.518),
                       "wordnet": (0.361, 0.483, 0.581, 1940, 0.440),
                       "nell": (0.447, 0.564, 0.695, 965, 0.530)}
    vanilla_seed_options = {"freebase": 4089853924, "wordnet": 1919180054, "nell": 3192206669}
    random_seed_options = {"freebase": 123456789, "wordnet": 627997250, "nell": 3192206669}

    num_leiden_samples, num_random_samples = 1000, 100
    N, K, V_size, V_star_size, total_eval_embeddings = dict(), dict(), dict(), dict(), dict()
    datasets_resolution, datasets_cut_size, datasets_modularity = dict(), dict(), dict()
    datasets_speed_up, datasets_bulk_up = dict(), dict()
    datasets_test_edges_communities, datasets_com_test_edges = dict(), dict()
    datasets_test_edges_communities_extra, datasets_com_test_edges_extra = dict(), dict()
    leiden_resolution_scales = np.logspace(-5, 5, num_leiden_samples)
    leiden_resolution_scales_extra = np.array([1e-1, 1e0, 1e1, 1e2, 1e3, 1e4])
    plot_data_leiden = []
    plot_data_leiden_extra = []
    plot_data_metis = []
    plot_data_random = []
    for dataset_key, dataset_name in datasets_names.items():
        with open(f"graph_completion/configs/{dataset_key}.yml", "r", encoding="utf-8") as config_file:
            dataset_conf = ExperimentHpars.from_dict(yaml.safe_load(config_file)).make()
        reproduce(vanilla_seed_options[dataset_key])
        dataset_loader = LoaderHpars.from_dict(dataset_conf.loader_hpars).make()
        dataset_loader.load_graph(vanilla_seed_options[dataset_key], "cpu",
                                  dataset_conf.val_size, dataset_conf.test_size,
                                  dataset_conf.community_method, dataset_conf.leiden_resolution)
        N[dataset_key] = len(dataset_loader.test_edge_data)
        K[dataset_key] = dataset_loader.num_communities
        V_size[dataset_key] = dataset_loader.num_nodes
        V_star_size[dataset_key] = dataset_loader.inter_community_map.max()
        c_s = dataset_loader.communities[dataset_loader.test_edge_data.s]
        c_t = dataset_loader.communities[dataset_loader.test_edge_data.t]

        datasets_test_edges_communities[dataset_key] = c_t[np.argsort(dataset_loader.num_communities * c_s + c_t)]
        com_test_edges = dataset_loader.test_edge_data.assign(c_t=c_t).groupby("c_t").size().reindex(
            np.arange(dataset_loader.num_communities), fill_value=0
        ).values
        datasets_com_test_edges[dataset_key] = com_test_edges
        com_test_embeddings = dataset_loader.num_communities + dataset_loader.community_sizes
        total_eval_embeddings[dataset_key] = (com_test_edges * com_test_embeddings).sum()
        datasets_speed_up[dataset_key] = (N[dataset_key] * V_size[dataset_key]) / total_eval_embeddings[dataset_key]
        datasets_bulk_up[dataset_key] = (V_size[dataset_key] + K[dataset_key]
                                         + V_star_size[dataset_key] + 1) / V_size[dataset_key]
        print(f"Acceleration for {datasets_names[dataset_key]}: {round(datasets_speed_up[dataset_key], 4)}")
        print(f"Overparametrization for {datasets_names[dataset_key]}: {round(datasets_bulk_up[dataset_key], 4)}")

        subgraphing = Subgraphing(dataset_loader.graph, None, dataset_loader.num_nodes, "leiden",
                                  dataset_conf.leiden_resolution, 1, 250)
        subgraphing.recursive_updates(dataset_loader.dataset.node_data, dataset_loader.dataset.edge_data)
        subgraphing.compute_metrics()
        datasets_resolution[dataset_key] = subgraphing.community_resolution_disjoint
        datasets_cut_size[dataset_key] = subgraphing.metric_values['DisjointCommunityCutSize'][-1]
        datasets_modularity[dataset_key] = subgraphing.metric_values['DisjointCommunityModularity'][-1]
        leiden_resolution_values = subgraphing.community_resolution_disjoint * leiden_resolution_scales
        leiden_resolution_values_extra = subgraphing.community_resolution_disjoint * leiden_resolution_scales_extra
        for leiden_resolution in tqdm(leiden_resolution_values, "Validating Leiden resolution", leave=False):
            reproduce(dataset_conf.seed)
            subgraphing.community_resolution_disjoint = leiden_resolution
            subgraphing.recursive_updates(dataset_loader.dataset.node_data, dataset_loader.dataset.edge_data)
            subgraphing.compute_metrics()
            community_method_metrics, _ = calculate_community_method_metrics(subgraphing)
            plot_data_leiden.append((dataset_key, leiden_resolution, *community_method_metrics))
        for i, leiden_resolution in tqdm(enumerate(leiden_resolution_values_extra),
                                         "Validating Leiden resolution", total=6, leave=False):
            reproduce(vanilla_seed_options[dataset_key] if i == 1 else dataset_conf.seed)
            subgraphing.community_resolution_disjoint = leiden_resolution
            subgraphing.recursive_updates(dataset_loader.dataset.node_data, dataset_loader.dataset.edge_data)
            subgraphing.compute_metrics()
            (community_method_metrics,
             (test_edges_communities, com_test_edges)) = calculate_community_method_metrics(subgraphing)
            datasets_test_edges_communities_extra[
                (dataset_key, ("leiden", leiden_resolution))
            ] = test_edges_communities
            datasets_com_test_edges_extra[(dataset_key, ("leiden", leiden_resolution))] = com_test_edges
            plot_data_leiden_extra.append((dataset_key, leiden_resolution, *community_method_metrics))

        subgraphing.community_method = "metis"
        reproduce(dataset_conf.seed)
        subgraphing.recursive_updates(dataset_loader.dataset.node_data, dataset_loader.dataset.edge_data)
        subgraphing.compute_metrics()
        (community_method_metrics,
         (test_edges_communities, com_test_edges)) = calculate_community_method_metrics(subgraphing)
        datasets_test_edges_communities_extra[
            (dataset_key, ("metis", datasets_resolution[dataset_key]))
        ] = test_edges_communities
        datasets_com_test_edges_extra[(dataset_key, ("metis", datasets_resolution[dataset_key]))] = com_test_edges
        plot_data_metis.append((dataset_key, *community_method_metrics))

        subgraphing.community_method = "random"
        reproduce(random_seed_options[dataset_key])
        for sample_id in tqdm(range(num_random_samples), "Validating random community partition", leave=False):
            subgraphing.recursive_updates(dataset_loader.dataset.node_data, dataset_loader.dataset.edge_data)
            subgraphing.compute_metrics()
            (community_method_metrics,
             (test_edges_communities, com_test_edges)) = calculate_community_method_metrics(subgraphing)
            if sample_id == 0:
                datasets_test_edges_communities_extra[
                    (dataset_key, ("random", datasets_resolution[dataset_key]))
                ] = test_edges_communities
                datasets_com_test_edges_extra[
                    (dataset_key, ("random", datasets_resolution[dataset_key]))
                ] = com_test_edges
            plot_data_random.append((dataset_key, sample_id, *community_method_metrics))

    plot_data_leiden = pd.DataFrame(plot_data_leiden, columns=["Dataset", "LeidenResolution",
                                                               "StdCommunitySizes", "StdCommunityNumTestEdges",
                                                               "CutSize", "Modularity",
                                                               "Acceleration", "Overparametrization"])
    plot_data_leiden = plot_data_leiden.melt(id_vars=["Dataset", "LeidenResolution",
                                                      "StdCommunitySizes", "StdCommunityNumTestEdges",
                                                      "CutSize", "Modularity"],
                                             value_vars=["Acceleration", "Overparametrization"],
                                             var_name="Factor", value_name="Value")
    plot_data_leiden_extra = pd.DataFrame(plot_data_leiden_extra, columns=["Dataset", "LeidenResolution",
                                                                           "StdCommunitySizes",
                                                                           "StdCommunityNumTestEdges",
                                                                           "CutSize", "Modularity",
                                                                           "Acceleration",
                                                                           "Overparametrization"])
    plot_data_metis = pd.DataFrame(plot_data_metis, columns=["Dataset",
                                                             "StdCommunitySizes", "StdCommunityNumTestEdges",
                                                             "CutSize", "Modularity",
                                                             "Acceleration", "Overparametrization"])
    plot_data_random = pd.DataFrame(plot_data_random, columns=["Dataset", "SampleId",
                                                               "StdCommunitySizes", "StdCommunityNumTestEdges",
                                                               "CutSize", "Modularity",
                                                               "Acceleration", "Overparametrization"])

    factor_color_map = {factor_name: color for factor_name, color in zip(
        ["speed_up", "bulk_up"], sns.color_palette("Set1", n_colors=len(datasets_names), desat=0.75).as_hex()
    )}
    plt.figure(figsize=(3 * 11.7, 8.3))
    g = sns.relplot(x="LeidenResolution", y="Value", hue="Factor", col="Dataset", data=plot_data_leiden,
                    palette="Set1", linewidth=5, kind="line",
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"legend_out": True})
    for dataset_key, ax in g.axes_dict.items():
        ax.axvline(datasets_resolution[dataset_key], 0, sqrt(V_size[dataset_key]) / 2,
                   color="black", linestyle="dashed", lw=5)
        ax.set_title(datasets_names[dataset_key])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Leiden resolution")
        ax.set_ylabel("Factor value")
    for lh in g.legend.get_lines():
        lh.set_linewidth(5)
    g.tight_layout()
    plt.savefig("graph_completion/results/scalability_leiden_resolution.pdf", format="pdf")

    plt.figure(figsize=(3 * 11.7, 8.3))
    g = sns.relplot(x="CutSize", y="Value", hue="Factor", col="Dataset", data=plot_data_leiden,
                    palette="Set1", linewidth=5, kind="line",
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"sharex": False, "legend_out": True})
    for dataset_key, ax in g.axes_dict.items():
        ax.axvline(datasets_cut_size[dataset_key], 0, sqrt(V_size[dataset_key]) / 2,
                   color="black", linestyle="dashed", lw=5)
        metis_cut_size = plot_data_metis[plot_data_metis.Dataset == dataset_key].CutSize.iloc[0]
        random_cut_size = plot_data_random[plot_data_random.Dataset == dataset_key].CutSize
        random_speed_up = plot_data_random[plot_data_random.Dataset == dataset_key].Acceleration
        random_bulk_up = plot_data_random[plot_data_random.Dataset == dataset_key].Overparametrization
        ax.axvline(metis_cut_size, 0, sqrt(V_size[dataset_key]) / 2, color="gray", linestyle="dashed", lw=5)
        ax.errorbar(x=random_cut_size.mean(), y=random_speed_up.mean(),
                    xerr=random_cut_size.std(), yerr=random_speed_up.std(),
                    fmt=factor_color_map["speed_up"], ecolor=factor_color_map["speed_up"], marker="s",
                    markersize=20, elinewidth=5, capsize=20, capthick=5)
        ax.errorbar(x=random_cut_size.mean(), y=random_bulk_up.mean(),
                    xerr=random_cut_size.std(), yerr=random_bulk_up.std(),
                    fmt=factor_color_map["bulk_up"], ecolor=factor_color_map["bulk_up"], marker="s",
                    markersize=20, elinewidth=5, capsize=20, capthick=5)
        ax.set_title(datasets_names[dataset_key])
        ax.set_yscale("log")
        ax.set_xlabel("Cut size")
        ax.set_ylabel("Factor value")
    for lh in g.legend.get_lines():
        lh.set_linewidth(5)
    g.tight_layout()
    plt.savefig("graph_completion/results/scalability_leiden_cut_size.pdf", format="pdf")

    plt.figure(figsize=(3 * 11.7, 8.3))
    g = sns.relplot(x="Modularity", y="Value", hue="Factor", col="Dataset", data=plot_data_leiden,
                    palette="Set1", linewidth=5, kind="line",
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"legend_out": True})
    for dataset_key, ax in g.axes_dict.items():
        ax.axvline(datasets_modularity[dataset_key], 0, sqrt(V_size[dataset_key]) / 2,
                   color="black", linestyle="dashed", lw=5)
        metis_modularity = plot_data_metis[plot_data_metis.Dataset == dataset_key].Modularity.iloc[0]
        random_modularity = plot_data_random[plot_data_random.Dataset == dataset_key].Modularity
        random_speed_up = plot_data_random[plot_data_random.Dataset == dataset_key].Acceleration
        random_bulk_up = plot_data_random[plot_data_random.Dataset == dataset_key].Overparametrization
        ax.axvline(metis_modularity, 0, sqrt(V_size[dataset_key]) / 2, color="gray", linestyle="dashed", lw=5)
        ax.errorbar(x=random_modularity.mean(), y=random_speed_up.mean(),
                    xerr=random_modularity.std(), yerr=random_speed_up.std(),
                    fmt=factor_color_map["speed_up"], ecolor=factor_color_map["speed_up"], marker="s",
                    markersize=20, elinewidth=5, capsize=20, capthick=5)
        ax.errorbar(x=random_modularity.mean(), y=random_bulk_up.mean(),
                    xerr=random_modularity.std(), yerr=random_bulk_up.std(),
                    fmt=factor_color_map["bulk_up"], ecolor=factor_color_map["bulk_up"], marker="s",
                    markersize=20, elinewidth=5, capsize=20, capthick=5)
        ax.set_title(datasets_names[dataset_key])
        ax.set_yscale("log")
        ax.set_xlabel("Modularity")
        ax.set_ylabel("Factor value")
    for lh in g.legend.get_lines():
        lh.set_linewidth(5)
    g.tight_layout()
    plt.savefig("graph_completion/results/scalability_leiden_modularity.pdf", format="pdf")

    train_logs = [pd.read_csv(f"{run_folder}/train_log.txt", sep="\t", encoding="utf-8", header=None)
                  for dataset in datasets_names for run_folder in glob(f"graph_completion/results/{dataset}/runs/*")
                  if exists(f"{run_folder}/train_log.txt")]
    test_logs = [pd.read_csv(f"{run_folder}/test_log.txt", sep="\t", encoding="utf-8",
                             header=None, index_col=None)
                 for dataset in datasets_names for run_folder in glob(f"graph_completion/results/{dataset}/runs/*")
                 if exists(f"{run_folder}/test_log.txt")]
    test_logs_extra = [pd.read_csv(f"{run_folder}/test_log.txt", sep="\t", encoding="utf-8",
                                   header=None, index_col=None)
                       for dataset in datasets_names for run_folder in
                       glob(f"graph_completion/results_extra/{dataset}/*")
                       if exists(f"{run_folder}/test_log.txt")]
    test_ranks = [pd.read_csv(f"{run_folder}/test_ranks.txt", sep="\t", encoding="utf-8",
                              header=None, index_col=None, dtype={10: "string"}).reset_index()
                  for dataset in datasets_names for run_folder in glob(f"graph_completion/results/{dataset}/runs/*")
                  if exists(f"{run_folder}/test_ranks.txt")]
    leiden_resolution_scales_extra[0], leiden_resolution_scales_extra[1] = (
        leiden_resolution_scales_extra[1], leiden_resolution_scales_extra[0]
    )
    test_ranks_extra = [pd.read_csv(f"{run_folder}/test_ranks.txt", sep="\t", encoding="utf-8",
                                    header=None, index_col=None, dtype={11: "string"}).reset_index().assign(
        CommunityMethod="metis" if i == 6 else ("random" if i == 7 else "leiden"),
        LeidenResolution=(datasets_resolution[dataset]
                          * leiden_resolution_scales_extra[i]) if i < 6 else datasets_resolution[dataset]
    )
        for dataset in datasets_names for i, run_folder in
        enumerate(glob(f"graph_completion/results_extra/{dataset}/*"))
        if exists(f"{run_folder}/test_ranks.txt")]
    train_logs = pd.concat(train_logs)
    test_logs = pd.concat(test_logs, ignore_index=True)
    test_logs_extra = pd.concat(test_logs_extra, ignore_index=True)
    test_ranks = pd.concat(test_ranks, ignore_index=True)
    test_ranks_extra = pd.concat(test_ranks_extra, ignore_index=True)

    train_logs.columns = ["Seed", "MiniBatchSize", "LearningRate", "Algorithm", "LeidenResolution",
                          "EmbeddingDim", "LossMargin", "Dataset", "NumNodes", "NumNodeTypes", "NumRelations",
                          "NumCommunities", "NumNegativeSamples", "EdgeDensity", "NodeTypeAssortativity",
                          "GiantWCC", "GiantSCC", "AveragePathLength", "Diameter", "AverageClustering",
                          "TrainComLoss", "TrainNodeLoss", "TrainLoss", "TrainTime",
                          "ValComLoss", "ValNodeLoss", "ValLoss",
                          "ComAccuracy", "Accuracy", "ComPrecision", "Precision", "ComRecall", "Recall", "ComF1", "F1",
                          "ComROC-AUC", "ROC-AUC", "ComAP", "AP", "Patience"]
    train_logs = train_logs.reset_index().rename(columns={"index": "Batch"})
    train_logs = train_logs.assign(
        NumSamples=(train_logs.Batch + 1) * train_logs.MiniBatchSize * (train_logs.NumNegativeSamples + 1)
    )
    train_logs.loc[train_logs.Algorithm.isin(["distmult", "complex"]), "NumSamples"] *= 10
    test_logs.columns = ["Checkpoint", "Seed", "MiniBatchSize", "LearningRate", "Algorithm", "LeidenResolution",
                         "EmbeddingDim", "LossMargin", "Dataset", "NumNodes", "NumNodeTypes", "NumRelations",
                         "NumCommunities", "NumNegativeSamples", "EdgeDensity", "NodeTypeAssortativity",
                         "GiantWCC", "GiantSCC", "AveragePathLength", "Diameter", "AverageClustering",
                         "TestComLoss", "TestNodeLoss", "TestLoss",
                         "ComAccuracy", "Accuracy", "ComPrecision", "Precision", "ComRecall", "Recall", "ComF1", "F1",
                         "ComROC-AUC", "ROC-AUC", "ComAP", "AP", "ComHits@1", "NodeHits@1", "Hits@1",
                         "ComHits@3", "NodeHits@3", "Hits@3", "ComHits@10", "NodeHits@10", "Hits@10",
                         "ComMR", "NodeMR", "MR", "ComMRR", "NodeMRR", "MRR", "TestTime"]
    test_logs = test_logs.assign(Value="COINs")
    test_logs_extra.columns = ["Checkpoint", "Seed", "MiniBatchSize", "LearningRate",
                               "Algorithm", "CommunityMethod", "LeidenResolution",
                               "EmbeddingDim", "LossMargin", "Dataset", "NumNodes", "NumNodeTypes", "NumRelations",
                               "NumCommunities", "NumNegativeSamples", "EdgeDensity", "NodeTypeAssortativity",
                               "GiantWCC", "GiantSCC", "AveragePathLength", "Diameter", "AverageClustering",
                               "TestComLoss", "TestNodeLoss", "TestLoss",
                               "ComAccuracy", "Accuracy", "ComPrecision", "Precision", "ComRecall", "Recall",
                               "ComF1", "F1", "ComROC-AUC", "ROC-AUC", "ComAP", "AP",
                               "ComHits@1", "NodeHits@1", "Hits@1", "ComHits@3", "NodeHits@3", "Hits@3",
                               "ComHits@10", "NodeHits@10", "Hits@10", "ComMR", "NodeMR", "MR",
                               "ComMRR", "NodeMRR", "MRR", "TestTime"]
    test_logs_extra = test_logs_extra.assign(Value="COINs")
    test_ranks.columns = ["SampleId", "NumCommunities", "FilteredCommunities",
                          "CommunitySize", "FilteredCommunitySize",
                          "ComRank", "NodeRank", "Rank", "Dataset", "Algorithm", "Seed", "Checkpoint"]
    test_ranks.SampleId = test_ranks.apply(lambda row: row["SampleId"] % N[row["Dataset"]], axis=1)
    test_ranks = test_ranks.assign(NumNodes=test_ranks.Dataset.map(V_size), NumEdges=test_ranks.Dataset.map(N))
    test_ranks_extra.columns = ["SampleId", "NumCommunities", "FilteredCommunities",
                                "CommunitySize", "FilteredCommunitySize",
                                "ComRank", "NodeRank", "Rank", "Query", "Dataset", "Algorithm", "Seed", "Checkpoint",
                                "CommunityMethod", "LeidenResolution"]
    test_ranks_extra.SampleId = test_ranks_extra.apply(lambda row: row["SampleId"] % N[row["Dataset"]], axis=1)
    test_ranks_extra = test_ranks_extra.assign(NumNodes=test_ranks_extra.Dataset.map(V_size),
                                               NumEdges=test_ranks_extra.Dataset.map(N))

    test_logs_baseline = []
    for dataset_key, dataset_name in datasets_names.items():
        for algorithm_key, algorithm_name in algs_names.items():
            test_logs_baseline.append((dataset_key, algorithm_key,
                                       *parse_baseline_metrics(dataset_key, algorithm_name)))
    test_logs_baseline = pd.DataFrame(test_logs_baseline, columns=["Dataset", "Algorithm",
                                                                   "Hits@1", "Hits@3", "Hits@10", "MR", "MRR"])
    test_logs_baseline = test_logs_baseline.assign(Value="Baseline", Seed=0)

    results_table = pd.concat((test_logs_baseline, test_logs))
    results_table = results_table.groupby(["Seed", "Dataset", "Algorithm", "Value"], as_index=False)[
        ["Hits@1", "Hits@3", "Hits@10", "MRR"]
    ].max()
    results_table_std = results_table.groupby(["Dataset", "Algorithm", "Value"])[
        ["Hits@1", "Hits@3", "Hits@10", "MRR"]
    ].std().round(decimals=3)
    results_table_std = results_table_std.reindex(list(datasets_names.keys()), level=0)
    results_table_std = results_table_std.reindex(list(algs_names.keys()), level=1)
    results_table_std = results_table_std.reindex(["COINs", ], level=2)
    results_table = results_table.groupby(["Dataset", "Algorithm", "Value"])[
        ["Hits@1", "Hits@3", "Hits@10", "MRR"]
    ].mean().round(decimals=3)
    results_table = results_table.reindex(list(datasets_names.keys()), level=0)
    results_table = results_table.reindex(list(algs_names.keys()), level=1)
    results_table = results_table.reindex(["Baseline", "COINs"], level=2)
    print(results_table)
    print(results_table_std)
    results_table_2 = test_logs.groupby(["Seed", "Dataset", "Algorithm"], sort=False)[
        ["ComHits@1", "NodeHits@1", "ComHits@3", "NodeHits@3", "ComHits@10", "NodeHits@10", "ComMRR", "NodeMRR"]
    ].max().reset_index()
    results_table_2_parts = []
    for metric in ["Hits@1", "Hits@3", "Hits@10", "MRR"]:
        results_table_2_part = results_table_2.melt(id_vars=["Seed", "Dataset", "Algorithm"],
                                                    value_vars=[f"Com{metric}", f"Node{metric}"],
                                                    var_name="Value",
                                                    value_name=metric)
        results_table_2_part.Value = results_table_2_part.Value.map({f"Com{metric}": "Community",
                                                                     f"Node{metric}": "Node"})
        results_table_2_part = results_table_2_part.set_index(["Seed", "Dataset", "Algorithm", "Value"])
        results_table_2_parts.append(results_table_2_part)
    results_table_2 = pd.concat(results_table_2_parts, axis="columns")
    results_table_std_2 = results_table_2.groupby(["Dataset", "Algorithm", "Value"])[
        ["Hits@1", "Hits@3", "Hits@10", "MRR"]
    ].std().round(decimals=3)
    results_table_std_2 = results_table_std_2.reindex(list(datasets_names.keys()), level=0)
    results_table_std_2 = results_table_std_2.reindex(list(algs_names.keys()), level=1)
    results_table_std_2 = results_table_std_2.reindex(["Community", "Node"], level=2)
    results_table_2 = results_table_2.groupby(["Dataset", "Algorithm", "Value"])[
        ["Hits@1", "Hits@3", "Hits@10", "MRR"]
    ].mean().round(decimals=3)
    results_table_2 = results_table_2.reindex(list(datasets_names.keys()), level=0)
    results_table_2 = results_table_2.reindex(list(algs_names.keys()), level=1)
    results_table_2 = results_table_2.reindex(["Community", "Node"], level=2)
    print(results_table_2)
    print(results_table_std_2)
    results_table_3 = test_logs.groupby(["Seed", "Dataset", "Algorithm"], sort=False)[
        ["ComAccuracy", "Accuracy", "ComF1", "F1", "ComROC-AUC", "ROC-AUC", "ComAP", "AP"]
    ].max().reset_index()
    results_table_3_parts = []
    for metric in ["Accuracy", "F1", "ROC-AUC", "AP"]:
        results_table_3_part = results_table_3.melt(id_vars=["Seed", "Dataset", "Algorithm"],
                                                    value_vars=[f"Com{metric}", metric],
                                                    var_name="Value",
                                                    value_name=metric)
        results_table_3_part.Value = results_table_3_part.Value.map({f"Com{metric}": "Community",
                                                                     metric: "Overall"})
        results_table_3_part = results_table_3_part.set_index(["Seed", "Dataset", "Algorithm", "Value"])
        results_table_3_parts.append(results_table_3_part)
    results_table_3 = pd.concat(results_table_3_parts, axis="columns")
    results_table_std_3 = results_table_3.groupby(["Dataset", "Algorithm", "Value"])[
        ["Accuracy", "F1", "ROC-AUC", "AP"]
    ].std().round(decimals=3)
    results_table_std_3 = results_table_std_3.reindex(list(datasets_names.keys()), level=0)
    results_table_std_3 = results_table_std_3.reindex(list(algs_names.keys()), level=1)
    results_table_std_3 = results_table_std_3.reindex(["Community", "Overall"], level=2)
    results_table_3 = results_table_3.groupby(["Dataset", "Algorithm", "Value"])[
        ["Accuracy", "F1", "ROC-AUC", "AP"]
    ].mean().round(decimals=3)
    results_table_3 = results_table_3.reindex(list(datasets_names.keys()), level=0)
    results_table_3 = results_table_3.reindex(list(algs_names.keys()), level=1)
    results_table_3 = results_table_3.reindex(["Community", "Overall"], level=2)
    print(results_table_3)
    print(results_table_std_3)

    test_logs = test_logs.loc[test_logs.Checkpoint == "best"]
    test_logs_baseline = test_logs_baseline.set_index(["Dataset", "Algorithm"])["MR"]
    test_ranks = test_ranks.loc[test_ranks.Checkpoint == "best"].drop(columns=["Checkpoint", ])
    test_ranks = test_ranks.assign(Community=test_ranks.apply(
        lambda row: datasets_test_edges_communities[row["Dataset"]][row["SampleId"]], axis=1
    ))
    test_ranks = test_ranks.groupby(
        ["Seed", "Dataset", "Algorithm", "Community"], sort=False, as_index=False
    )[["NumNodes", "NumCommunities", "CommunitySize", "NumEdges", "NodeRank", "Rank"]].mean()
    test_ranks = test_ranks.assign(
        Acceleration=test_ranks.NumNodes / (test_ranks.NumCommunities + test_ranks.CommunitySize),
        CommunityNumEdges=test_ranks.apply(
            lambda row: datasets_com_test_edges[row["Dataset"]][row["Community"]], axis=1
        )
    )
    test_ranks = test_ranks.assign(CommunityWeight=test_ranks.CommunityNumEdges / test_ranks.NumEdges)
    test_ranks = test_ranks.drop(columns=["CommunityNumEdges", "NumEdges"])
    test_ranks = test_ranks.set_index(["Community", "Seed", "Dataset", "Algorithm"])
    test_ranks = test_ranks.join(test_logs_baseline).reorder_levels(["Community", "Seed", "Dataset", "Algorithm"])
    test_ranks = test_ranks.assign(
        RelativeError=-(test_ranks.MR - test_ranks.Rank) / test_ranks.MR,
        NodeRelativeError=-(test_ranks.MR - test_ranks.NodeRank) / test_ranks.MR,
        Expectation=test_ranks.CommunityWeight * (test_ranks.NumCommunities
                                                  + test_ranks.CommunitySize) * test_ranks.Rank,
        NodeExpectation=test_ranks.CommunityWeight * (test_ranks.NumCommunities
                                                      + test_ranks.CommunitySize) * test_ranks.NodeRank
    )
    test_ranks = test_ranks.drop(columns=["NumNodes", "NumCommunities", "CommunitySize", "MR"])

    print(f"Median acceleration: {round(test_ranks.Acceleration.median(), 4)} "
          f"(+/- {round(test_ranks.Acceleration.mad(), 4)})")
    print(f"Median relative error: {round(test_ranks.RelativeError.median(), 4)} "
          f"(+/- {round(test_ranks.RelativeError.mad(), 4)})")

    test_ranks = test_ranks.assign(
        Statistic=test_ranks.CommunityWeight * (1 + test_ranks.RelativeError) / test_ranks.Acceleration,
        NodeStatistic=test_ranks.CommunityWeight * (1 + test_ranks.NodeRelativeError) / test_ranks.Acceleration
    )
    test_ranks = test_ranks.drop(columns=["CommunityWeight", "Rank", "NodeRank",
                                          "Acceleration", "RelativeError", "NodeRelativeError"])
    test_ranks = test_ranks.groupby(level=[1, 2, 3]).sum()
    test_logs_baseline = test_logs_baseline.reset_index()
    test_logs_baseline = test_logs_baseline.assign(
        BaselineExpectation=test_logs_baseline.Dataset.map(V_size) * test_logs_baseline.MR
    ).set_index(["Dataset", "Algorithm"])
    test_ranks = test_ranks.join(test_logs_baseline).reset_index()

    plot_data = test_ranks.rename(
        columns={"BaselineExpectation": "Baseline", "Expectation": "Overall", "NodeExpectation": "Node"}
    ).melt(id_vars=["Seed", "Dataset", "Algorithm", "Baseline"],
           value_vars=["Overall", "Node"],
           var_name="Value", value_name="COINs")
    plot_data_std = plot_data.groupby(["Dataset", "Algorithm", "Value"])[["COINs", "Baseline"]].std()
    plot_data_std = plot_data_std.reindex(list(datasets_names.keys()), level=0)
    plot_data_std = plot_data_std.reindex(list(algs_names.keys()), level=1)
    plot_data_std = plot_data_std.reindex(["Overall", "Node"], level=2)
    plot_data_std = plot_data_std.reset_index()
    plot_data_std.Dataset = plot_data_std.Dataset.map(datasets_names)
    plot_data_std.Algorithm = plot_data_std.Algorithm.map(algs_names)
    plot_data = plot_data.groupby(["Dataset", "Algorithm", "Value"])[["COINs", "Baseline"]].mean()
    plot_data = plot_data.reindex(list(datasets_names.keys()), level=0)
    plot_data = plot_data.reindex(list(algs_names.keys()), level=1)
    plot_data = plot_data.reindex(["Overall", "Node"], level=2)
    plot_data = plot_data.reset_index()
    plot_data.Dataset = plot_data.Dataset.map(datasets_names)
    plot_data.Algorithm = plot_data.Algorithm.map(algs_names)
    x_boundary = np.logspace(5, 8, 10000, endpoint=False)
    y_boundary = x_boundary

    plt.figure(figsize=(3 * 11.7, 2 * 8.3))
    g = sns.relplot(x="COINs", y="Baseline", hue="Dataset", style="Algorithm",
                    row="Value", data=plot_data,
                    palette="Set1", s=1000, kind="scatter",
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"legend_out": True})
    dataset_color_map = {dataset_name: color for (_, dataset_name), color in zip(
        datasets_names.items(), sns.color_palette("Set1", n_colors=len(datasets_names), desat=0.75)
    )}
    for value, ax in g.axes_dict.items():
        ax.set_title(fr"{value} $\mathbb{{E}}[T H]$")
        ax.fill_between(x_boundary, y_boundary, 1e10, color="gray", alpha=0.25)
        ax.plot(x_boundary, y_boundary, color="black", linestyle="dashed", lw=5)
        mean_data = plot_data[plot_data.Value == value]
        std_data = plot_data_std[plot_data_std.Value == value]
        error_bar_colors = [dataset_color_map[dataset] for dataset in std_data.Dataset]
        for x, y, xerr, yerr, ecolor in zip(mean_data.COINs, mean_data.Baseline,
                                            std_data.COINs, std_data.Baseline, error_bar_colors):
            ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, fmt="none", ecolor=ecolor, elinewidth=5, capsize=20, capthick=5)
        ax.set_xlim(1e5, 1e8)
        ax.set_ylim(1e5, 1e10)
        ax.set_xscale("log")
        ax.set_yscale("log")
    for lh in g.legend.legendHandles:
        lh.set_sizes([1000])
    g.tight_layout()
    plt.savefig("graph_completion/results/feasibility.pdf", format="pdf")

    plot_data = test_ranks.rename(
        columns={"Statistic": "Overall", "NodeStatistic": "Node"}
    ).melt(id_vars=["Seed", "Dataset", "Algorithm"],
           value_vars=["Overall", "Node"],
           var_name="Value", value_name="Statistic")
    plot_data = plot_data.set_index(["Dataset", "Algorithm", "Value", "Seed"])
    plot_data = plot_data.reindex(list(datasets_names.keys()), level=0)
    plot_data = plot_data.reindex(list(algs_names.keys()), level=1)
    plot_data = plot_data.reindex(["Overall", "Node"], level=2)
    plot_data = plot_data.reset_index()
    plot_data.Dataset = plot_data.Dataset.map(datasets_names)
    plot_data.Algorithm = plot_data.Algorithm.map(algs_names)
    x_boundary = np.linspace(-0.5, 4.5, 10000, endpoint=False)

    plt.figure(figsize=(3 * 11.7, 2 * 8.3))
    g = sns.catplot(x="Algorithm", y="Statistic", hue="Dataset",
                    row="Value", data=plot_data,
                    palette="Set1", kind="bar", ci="sd",
                    capsize=0.25, errwidth=5, linewidth=5,
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"legend_out": True})
    dataset_color_map = {dataset_name: color for (_, dataset_name), color in zip(
        datasets_names.items(), sns.color_palette("Set1", n_colors=len(datasets_names), desat=0.75)
    )}
    for value, ax in g.axes_dict.items():
        ax.set_title(f"{value} value")
        plt.setp(ax.collections, sizes=[500])
        ax.fill_between(x_boundary, np.zeros_like(x_boundary), 1, color="gray", alpha=0.25)
        ax.plot(x_boundary, np.ones_like(x_boundary), color="black", linestyle="dashed", lw=5)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_ylim(1e-3, 5)
        ax.set_yscale("log")
        ax.set_ylabel(r"$\sum_{k=1}^{K}{\frac{|E_k^{\mathrm{test}}|}{N} \frac{1 + \varepsilon_k}{A_k}}$")
    for lh in g.legend.get_lines():
        lh.set_linewidth(5)
    g.tight_layout()
    plt.savefig("graph_completion/results/feasibility_statistic.pdf", format="pdf")

    test_ranks_extra = test_ranks_extra.loc[test_ranks_extra.Checkpoint == "best"].drop(
        columns=["Checkpoint", ]).astype({"LeidenResolution": float})
    test_ranks_extra = test_ranks_extra.assign(Community=test_ranks_extra.apply(
        lambda row: datasets_test_edges_communities_extra[
            (row["Dataset"], (row["CommunityMethod"], row["LeidenResolution"]))
        ][row["SampleId"]], axis=1
    ))
    test_ranks_extra = test_ranks_extra.groupby(
        ["Seed", "Dataset", "Algorithm", "CommunityMethod", "LeidenResolution", "Community"], sort=False, as_index=False
    )[["NumNodes", "NumCommunities", "CommunitySize", "NumEdges", "NodeRank", "Rank"]].mean()
    test_ranks_extra = test_ranks_extra.assign(
        Acceleration=test_ranks_extra.NumNodes / (test_ranks_extra.NumCommunities + test_ranks_extra.CommunitySize),
        CommunityNumEdges=test_ranks_extra.apply(
            lambda row: datasets_com_test_edges_extra[
                (row["Dataset"], (row["CommunityMethod"], row["LeidenResolution"]))
            ][row["Community"]], axis=1
        )
    )
    test_ranks_extra = test_ranks_extra.assign(
        CommunityWeight=test_ranks_extra.CommunityNumEdges / test_ranks_extra.NumEdges
    )
    test_ranks_extra = test_ranks_extra.drop(columns=["CommunityNumEdges", "NumEdges"])
    test_ranks_extra = test_ranks_extra.set_index(["Community", "Seed", "Dataset", "Algorithm",
                                                   "CommunityMethod", "LeidenResolution"])
    test_ranks_extra = test_ranks_extra.join(test_logs_baseline).reorder_levels(["Community", "Seed",
                                                                                 "Dataset", "Algorithm",
                                                                                 "CommunityMethod", "LeidenResolution"])
    test_ranks_extra = test_ranks_extra.assign(
        RelativeError=-(test_ranks_extra.MR - test_ranks_extra.Rank) / test_ranks_extra.MR,
        NodeRelativeError=-(test_ranks_extra.MR - test_ranks_extra.NodeRank) / test_ranks_extra.MR,
        Expectation=test_ranks_extra.CommunityWeight * (
                test_ranks_extra.NumCommunities + test_ranks_extra.CommunitySize
        ) * test_ranks_extra.Rank,
        NodeExpectation=test_ranks_extra.CommunityWeight * (
                test_ranks_extra.NumCommunities + test_ranks_extra.CommunitySize
        ) * test_ranks_extra.NodeRank
    )
    test_ranks_extra = test_ranks_extra.drop(columns=["NumNodes", "NumCommunities", "CommunitySize", "MR"])
    test_ranks_extra = test_ranks_extra.assign(
        Statistic=test_ranks_extra.CommunityWeight * (
                1 + test_ranks_extra.RelativeError
        ) / test_ranks_extra.Acceleration,
        NodeStatistic=test_ranks_extra.CommunityWeight * (
                1 + test_ranks_extra.NodeRelativeError
        ) / test_ranks_extra.Acceleration
    )
    test_ranks_extra = test_ranks_extra.drop(columns=["CommunityWeight", "Rank", "NodeRank",
                                                      "Acceleration", "RelativeError", "NodeRelativeError"])
    test_ranks_extra = test_ranks_extra.groupby(level=[1, 2, 3, 4, 5]).sum().reset_index()

    plot_data = test_ranks_extra.rename(
        columns={"Statistic": "Overall", "NodeStatistic": "Node"}
    ).melt(id_vars=["Seed", "Dataset", "Algorithm", "CommunityMethod", "LeidenResolution"],
           value_vars=["Overall", "Node"],
           var_name="Value", value_name="Statistic").astype({"LeidenResolution": float})
    plot_data = pd.concat((
        plot_data.loc[plot_data.CommunityMethod == "leiden"].merge(plot_data_leiden_extra,
                                                                   on=["Dataset", "LeidenResolution"]),
        plot_data.loc[plot_data.CommunityMethod == "metis"].merge(plot_data_metis, on="Dataset"),
        plot_data.loc[plot_data.CommunityMethod == "random"].merge(
            plot_data_random.groupby("Dataset", as_index=False).mean(),
            on="Dataset")
    ), ignore_index=True)

    plot_data = plot_data.set_index(
        ["Dataset", "CommunityMethod", "LeidenResolution", "Value"]
    ).sort_index().drop(columns=["Algorithm", ])
    plot_data = plot_data.sort_index(level=[0, 1, 2])
    plot_data = plot_data.reindex(list(datasets_names.keys()), level=0)
    plot_data = plot_data.reindex(["leiden", "metis", "random"], level=1)
    plot_data = plot_data.reindex(["Overall", "Node"], level=3)
    plot_data = plot_data.reset_index()
    plot_data.Dataset = plot_data.Dataset.map(datasets_names)
    plot_data.CommunityMethod = plot_data.CommunityMethod.map(
        {"leiden": "Leiden", "metis": "METIS", "random": "Random"})
    plot_data = plot_data.rename(columns={"CommunityMethod": "Algorithm"})
    x_boundary = np.linspace(0, 4e5, 10000, endpoint=False)

    plt.figure(figsize=(3 * 11.7, 2 * 8.3))
    g = sns.relplot(x="CutSize", y="Statistic", hue="Dataset", style="Algorithm",
                    row="Value", data=plot_data,
                    palette="Set1", kind="scatter", s=1000,
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"sharex": False, "legend_out": True})
    for value, ax in g.axes_dict.items():
        ax.set_title(f"{value} value")
        leiden_curves = plot_data[(plot_data.Algorithm == "Leiden") & (plot_data.Value == value)]
        for i in range(len(leiden_curves) - 1):
            dataset_1, dataset_2 = leiden_curves.iloc[i].Dataset, leiden_curves.iloc[i + 1].Dataset
            if dataset_1 != dataset_2:
                continue
            ax.annotate("", xytext=(leiden_curves.iloc[i].CutSize, leiden_curves.iloc[i].Statistic),
                        xy=(leiden_curves.iloc[i + 1].CutSize, leiden_curves.iloc[i + 1].Statistic),
                        arrowprops={"width": 5, "headwidth": 20, "headlength": 20,
                                    "color": dataset_color_map[dataset_1]})
        ax.fill_between(x_boundary, np.zeros_like(x_boundary), 1, color="gray", alpha=0.25)
        ax.plot(x_boundary, np.ones_like(x_boundary), color="black", linestyle="dashed", lw=5)
        ax.set_ylim(1e-6, 5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cut size")
        ax.set_ylabel(r"$\sum_{k=1}^{K}{\frac{|E^{\mathrm{test}}_{k}|}{N} \frac{1+\varepsilon_k}{A_k}}$")
    for lh in g.legend.legendHandles:
        lh.set_sizes([1000])
    g.tight_layout()
    plt.savefig("graph_completion/results/feasibility_extra.pdf", format="pdf")

    plot_data_train = train_logs[
        ["Seed", "Dataset", "Algorithm", "NumSamples", "TrainComLoss", "TrainNodeLoss", "TrainLoss"]
    ]
    plot_data_train = plot_data_train.rename(
        columns={"TrainComLoss": "Community", "TrainNodeLoss": "Node", "TrainLoss": "Overall"}
    ).melt(id_vars=["Seed", "Dataset", "Algorithm", "NumSamples"], value_vars=["Community", "Node", "Overall"],
           var_name="Value", value_name="Loss").assign(Subset="Training")
    plot_data_valid = train_logs[
        ["Seed", "Dataset", "Algorithm", "NumSamples", "ValComLoss", "ValNodeLoss", "ValLoss"]
    ]
    plot_data_valid = plot_data_valid.rename(
        columns={"ValComLoss": "Community", "ValNodeLoss": "Node", "ValLoss": "Overall"}
    ).melt(id_vars=["Seed", "Dataset", "Algorithm", "NumSamples"], value_vars=["Community", "Node", "Overall"],
           var_name="Value", value_name="Loss").assign(Subset="Validation")
    plot_data = pd.concat((plot_data_train, plot_data_valid), ignore_index=True)
    plot_data = plot_data.set_index(["Dataset", "Algorithm", "NumSamples", "Value", "Subset", "Seed"])
    plot_data = plot_data.reindex(list(datasets_names.keys()), level=0)
    plot_data = plot_data.reindex(list(algs_names.keys()), level=1)
    plot_data = plot_data.reset_index()

    plt.figure(figsize=(5 * 11.7, 2 * 8.3))
    g = sns.relplot(x="NumSamples", y="Loss", hue="Value",
                    row="Subset", col="Algorithm", data=plot_data,
                    palette="Set1", linewidth=5, kind="line", ci="sd",
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"sharex": False, "sharey": False, "legend_out": True})
    for (subset, algorithm_key), ax in g.axes_dict.items():
        ax.set_title(f"{algs_names[algorithm_key]} {subset}")
        ax.set_xscale("log")
        if algorithm_key in ["transe", "rotate"]:
            ax.set_xlim(right=2e8)
        ax.set_xlabel("Number of training samples")
    for lh in g.legend.get_lines():
        lh.set_linewidth(5)
    g.tight_layout()
    plt.savefig("graph_completion/results/convergence.pdf", format="pdf")
