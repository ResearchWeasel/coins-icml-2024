"""
Main script.
"""
import argparse
from copy import deepcopy
from pprint import pprint
from traceback import format_exc
from typing import List

import yaml

from graph_completion.experiments import ExperimentHpars
from graph_completion.graphs.load_graph import Loader
from graph_completion.utils import get_random_seed


def get_validation_configs(base_conf: dict) -> List[dict]:
    new_configs = []

    community_method = base_conf["community_method"] if "community_method" in base_conf else "leiden"
    if "leiden_resolution" in base_conf:
        leiden_resolution = base_conf["leiden_resolution"]
    else:
        dataset = Loader.datasets[base_conf["loader_hpars"]["dataset_name"]]
        dataset.load_from_disk()
        leiden_resolution = 1 / len(dataset.node_data)
        dataset.unload_from_memory()
    leiden_resolution_options = [leiden_resolution * factor for factor in [1e-1, 1e1, 1e2, 1e3, 1e4]]
    random_seed_options = [get_random_seed() for _ in range(5)]

    if community_method == "leiden":
        for leiden_resolution in leiden_resolution_options:
            new_config = deepcopy(base_conf)
            new_config["leiden_resolution"] = leiden_resolution
            new_configs.append(new_config)
    elif community_method == "random":
        for random_seed in random_seed_options:
            new_config = deepcopy(base_conf)
            new_config["seed"] = random_seed
            new_configs.append(new_config)
    else:
        new_configs.append(base_conf)

    return new_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate model.')
    parser.add_argument("-cf", "--config_file", metavar="config_file", type=str, required=True,
                        help="Path to the YAML config file containing the parameter settings.")
    parser.add_argument("-ns", "--num_seeds", metavar="num_seeds", type=int, required=False, default=1,
                        help="Number of different random seeds per parameter configuration.")
    parser.add_argument("-v", "--validate", metavar="validate", type=bool, required=False, default=False,
                        help="Whether to run hyperparameter validation.")
    args = parser.parse_args()
    config_filepath, num_seeds, validate = args.config_file, args.num_seeds, args.validate
    with open(config_filepath, "r", encoding="utf-8") as config_file:
        base_conf = yaml.safe_load(config_file)

    experiment_configs = get_validation_configs(base_conf) if validate else [base_conf, ]
    experiment_configs_seeds = []
    for experiment_config in experiment_configs:
        if num_seeds == 1:
            experiment_configs_seeds.append(experiment_config)
            continue
        for _ in range(num_seeds):
            experiment_config_seed = deepcopy(experiment_config)
            experiment_config_seed["seed"] = get_random_seed()
            experiment_configs_seeds.append(experiment_config_seed)
    for experiment_config in experiment_configs_seeds:
        experiment_hpars = ExperimentHpars.from_dict(experiment_config)
        pprint(experiment_config)
        experiment = experiment_hpars.make()
        try:
            experiment.main()
        except BaseException as error:
            with open("crash_log.txt", mode="w+", encoding="utf-8") as crash_log_file:
                crash_log_file.write(format_exc())
