# Co-occurrence Graph ATtention autoEncoder (CoGATE)

## Introduction

This is the code repository for the Co-occurrence Graph ATtention autoEncoder project. This readme file will walk through the following components:

+ [Dependency Requirement](#dependency-requirement)
+ [Dataset](#dataset)
+ [Configurations](#configurations)
+ [Automation Logic](#automation-logic)
+ [Usage](#usage)

## Dependency Requirement

+ TensorFlow (<=1.14.0)
+ PyTorch
+ NetworkX (>=2.0)
+ GenSim
+ openpyxl
+ [OpenNE: An Open-Source Package for Network Embedding](https://github.com/thunlp/OpenNE)
+ [GEM: Graph Embedding Methods](https://github.com/palash1992/GEM)

## Dataset

### Requirement

Please format and place the following file in the corresponding folder: [node file](#node-file), [edge list](#edge-list), [feature matrix](#nodal-features).

### Node File

*Location and Name*: Place the node file under the `./data/<org>/` folder with the name of `'<dataset>_nodes.csv'`.

*Format*: Each line of the file is a userID or repoID. NodeID will be assigned to each userID/repoID in the given order.

```text
Example:
Filename: './data/<org>/user_nodes.csv'
3985485 (NodeID = 0)
8279795 (NodeID = 1)
8296581 (NodeID = 2)
...     (NodeID = ...)
```

### Edge List

*Location and Name*: Place the edge list under the `./data/<org>/` folder with the name of `'<dataset>.edgelist'` or `'<dataset>commits.edgelist'` (`dataset` can be "user" or "repo", use the latter one when constructing the graph using commits).

*Format*: An un-directional edge between nodes `a` and `b` can be denoted with `a<space>b` or `b<space>a`. Each edge takes a new line. If the graph is weighted, each edge can be denoted as `a<space>b<space>w`.

```text
Example 1 (un-weighted, constructed using commits):
Filename: './data/<org>/usercommits.edgelist'
0 1
1 2
3 1
. .
```

```text
Example 2 (weighted):
Filename: './data/<org>/user.edgelist'
0 1 1.0
1 2 0.5
3 1 0.785
. . .
```

### Nodal Features

*Location and Name*: Nodal features are stored under the `./data/<org>/feature_matrices/` folder titled `'<dataset>_<random names>.csv'`.

*Format*: For `d`-dimension nodal features, each row has `d+1` values, with userID/repoID followed by `d` features.

```csv
Example:
Filename: './data/<org>/feature_matrices/user_test.csv'
3985485, 0.25, 0.35, 0.41, ...
8279795, 0.18, 0.36, 0.24, ...
...
...
```

## Configurations

All experiment configurations on graph embedding (GE) models and clustering algorithms are specified in `./graph_embedding_config.py`.

+ Change the configuration file before executing a new experiment.
+ Backup embeddings and results from the previous experiment.

## Automation Logic

The `./gva.py` script (usage [here](#usage)) automates the following steps:

1. [Preprocessing feature files](#step-1-preprocessing)
2. [Building and evaluating node embeddings](#step-2-building-and-evaluating-node-embedding)
3. [Exporiting results and Saving T-SNE plots](#step-3-export-results)

### Step 1. Preprocessing

The `./preprocess_data.py` script parse nodal feature CSVs and generate corresponding `.features` files under the `./data/<org>/` folder.

### Step 2. Building and Evaluating Node Embedding

The `./evaluation.py` script builds node embeddings for the selected datset and evaluate the quality of generated embeddings.

### Step 3. Exporting Results

The `./export_clustering_result.py` and `./plot_tsne.py` scripts exports experiment results to the following folders:

+ Dataset configuration: `./data/<org>/<dataset>_config.json`
+ Embeddings: `./embeddings/<org>/<dataset>/<feature file>/<GE model>.nv`  

    ```text
    #nodes #dim
    n0 e01 e02 e03 ... e0n
    n1 e11 e12 e13 ... e1n
    n2 ...
    .  ...
    ```

+ Runtime data: `./results/<org>/<dataset>/<feature file>/experiment-<n_cluster>.cache`
+ Evaluation results:
  + Mean Average Precision: `./results/<org>/<dataset>/MAP.xlsx`
  + KMeans results (cluster labels): `./results/<org>/<dataset>/KMeans_labels`
  + KMeans results (performance): `./results/<org>/<dataset>/KMeans_performance.xlsx`
+ t-SNE visualization of clustered embeddings: `./results/<org>/<dataset>/<tsne>/[<GE model>]<feature>_<clustering>_<# clusters>.svg`

## Usage

```text
usage: gva.py [-h] --org ORG --dataset {user,repo} --n_clusters N_CLUSTERS
              [--have_features HAVE_FEATURES]
              [--weighted_graph WEIGHTED_GRAPH] [--models MODELS]
              [--commit_edgelist COMMIT_EDGELIST] [--step {P,B,E,T,A}]

Automated GVA Processor.

optional arguments:
  -h, --help                        show this help message and exit
  --org             ORG             Organization of the analysis.
  --dataset         {user,repo}     Process 'user' or 'repo' dataset.
  --n_clusters      N_CLUSTERS      Comma delimited list input (e.g., 2,3,4,5,6), default=2.
  --have_features   HAVE_FEATURES   Whether the network has nodal features, default=True.
  --weighted_graph  WEIGHTED_GRAPH  Whether the edges are weighted, default=True.
  --models          MODELS          Comma delimited model names (e.g., TADW,GCAE,GATE),
                                    default=TADW,GCAE,GATE.
  --commit_edgelist COMMIT_EDGELIST Use edgelist constructed with commit info, default=False.
  --step            {P,B,E,T,A}     Perform a particular step ([P]reprocess, [B]uild embedding,
                                    [E]xport results, Plot [T]SNE) or [A]ll steps), default=A.
```

### *Example 1*

Execute the automated script for `CyVerse` on `user` dataset, edges are weighted and generated using commits. Evaluate the embeddings on `2, 4, 6, 8, 10` clusters.

```sh
python ./gva.py --org CyVerse --dataset user --comit_edgelist True --n_clusters 2,4,6,8,10
```

### *Example 2*

Execute the script for `tacc` on `repo` dataset, edges are weighted and not generated using commits. Preprocess the feature files only.

```sh
python ./gva.py --org tacc --dataset repo --step P
```

### *Example 3*

Assume some GE models did not produce valid embeddings for a particular feature file for Example 2, resulting in clustering errors (for 2,3,4,5,6 clusters).

1. Temporally move other feature files to a backup folder and keep the particular feature file(s) in the `feature_matrices` folder.

2. ```sh
   python ./gva.py --org tacc --dataset repo --step B --n_clusters 2,3,4,5,6
   ```

3. Move all feature files back to the `feature_matrices` folder.

4. ```sh
   python ./gva.py --org tacc --dataset repo --step E --n_clusters 2,3,4,5,6
   python ./gva.py --org tacc --dataset repo --step T --n_clusters 2,3,4,5,6
   ``
   