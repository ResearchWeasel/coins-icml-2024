# COINs: Knowledge Graph Inference with Model-based Acceleration

ICML 2024 conference submission

---------

## Abstract

We present a new method, called COINs for COmmunity INformed graph embeddings, to enhance the efficiency of knowledge graph models for link prediction and query answering. COINs uses a community-detection-based graph data augmentation and a two-step prediction pipeline: we first achieve node localization through community prediction, and subsequently, we further localize within the predicted community. We establish theoretical criteria to evaluate our method in our specific context and establish a direct expression of the reduction in time complexity. We empirically demonstrate an important scalability-performance trade-off where for a median relative per-sample drop in performance of 0.247, we obtain a median speed-up factor of 13.3038 on a single-CPU-GPU machine.

## Instructions

### Obtaining the code

Clone this repository by running:

`git clone https://github.com/ResearchWeasel/coins-icml-2024.git`

### Dependencies

The implementation requires version `3.6.13` of the Python programming language.
To install it and the dependent Python packages, we recommend having [Anaconda](https://www.anaconda.com/download), then
running the following commands from the main directory of the repository files:

1. `conda create --name coins python=3.6.13`
2. `conda activate coins`
3. `pip install -r requirements.txt`
4. `export PYTHONPATH='.'`

### Reproducing results

To regenerate the tables and figures provided in the paper, run the following command from the main directory of the
repository files:

`python graph_completion/plot.py`

The figure PDFs will be saved to the `graph_completion/results` directory.

To run end-to-end a COINs training and evaluation experiment from the paper, run the following command from the main
directory of the repository files:

- GPU:

  `CUDA_VISIBLE_DEVICES=<GPU ID> python graph_completion/main.py -cf='graph_completion/configs/<CONFIG FILENAME>.yml'`

- CPU (after setting the `device` config parameter to `cpu` in the YAML file):

  `python graph_completion/main.py -cf='graph_completion/configs/<CONFIG FILENAME>.yml'`

The experiment results will be saved to a directory in `graph_completion/results/<DATASET>/runs`.

----------

## Authors

- Anonymous
