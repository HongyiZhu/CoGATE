# Co-occurrence Graph ATtention autoEncoder (CoGATE)

## Introduction

This is the code repository for the Co-occurrence Graph ATtention autoEncoder project. This readme file will walk through the following components:

+ [Dependency Requirement](#dependency-requirement)
+ [Dataset](#dataset)
+ [Configurations](#configurations)
+ [Automation Logic](#automation-logic)
+ [Usage](#usage)

## Dependency Requirement

+ TensorFlow (>=2.0.0)
+ PyTorch
+ NetworkX (>=2.0)
+ GenSim
+ openpyxl
+ [OpenNE: An Open-Source Package for Network Embedding](https://github.com/thunlp/OpenNE)
  + Please use the [pytorch](https://github.com/thunlp/OpenNE/tree/pytorch) branch
  + [Git Clone Branch – How to Clone a Specific Branch](https://www.freecodecamp.org/news/git-clone-branch-how-to-clone-a-specific-branch/)
+ [GEM: Graph Embedding Methods](https://github.com/palash1992/GEM)

## Dataset

### Requirement

Please format and place the following file in the corresponding folder: [preprocessed sentence file](#preprocessed-sentence-documents), [full word list](#full-word-list), [edge list](#edge-list), [feature matrix](#nodal-features).

### Preprocessed Sentence Documents

*Location and Name*: Place the preprocessed sentence documents under the `./data/<dataset>/` folder with the name of `'<attackType>_<attackPlatform>_preprocessed_sent_document.pkl'`.

*Format*: The pickle dump file contains a tuple object `(sentences, documents)`. `sentences` comprises lists of tokenized documents and `documents` comprises a list of `gensim.TaggedDocument` objects. Each `TaggedDocument` object `documents[index]` is TaggedDocument(words=`sentences[index]`, tags=[`index`]).

### Full Word List

*Location and Name*: Place word list under the `./data/<dataset>/` folder with the name of `'<attackType>_<attackPlatform>_fullwordlist.pkl'`.

*Format*: The pickle dump file contains a list of unique words extracted from the preprocessed sentence documents. This dump is used to ensure a fixed wordID->nodeID association.

```text
Example:
Filename: './data/<dataset>/<attackType>_<attackPlatform>_fullwordlist.pkl'
Loaded list:
[
'Buffer',  # NodeID = 0
'Client',  # NodeID = 1
'Exploit', # NodeID = 2
...        # NodeID = ...
]
```

### Edge List

*Location and Name*: Place the edge list under the `./data/<dataset>/` folder with the name of `'<attackType>_<attackPlatform>.edgelist'`.

*Format*: An directional edge between nodes `a` and `b` can be denoted with `a<space>b`. Each edge takes a new line. If the graph is weighted, each edge can be denoted as `a<space>b<space>w`. *Duplicated edges are allowed*.

```text
Example 1 (un-weighted, constructed using commits):
Filename: './data/<dataset>/<attackType>_<attackPlatform>.edgelist'
0 1
1 2
3 1
. .
```

```text
Example 2 (weighted):
Filename: './data/<org>/<attackType>_<attackPlatform>.edgelist'
0 1 1.0
1 2 0.5
3 1 0.785
. . .
```

### Nodal Features

*Location and Name*: Nodal features are stored under the `./data/<dataset>/feature_matrices/<attackType>_<attackPlatform>/` folder titled `'<feature set name>.csv'`.

*Format*: The CSV file doesn't contain a header line. For `d`-dimension nodal features, each row has `d+1` values, with a word followed by `d` features.

```csv
Example:
Filename: './data/<dataset>/feature_matrices/<attackType>_<attackPlatform>/test.csv'
Buffer, 0.25, 0.35, 0.41, ...
Client, 0.18, 0.36, 0.24, ...
...
...
```

## Configurations

All experiment configurations on graph embedding (GE) models and clustering algorithms are specified in `./graph_embedding_config.py`.

+ Change the configuration file before executing a new experiment.
+ Backup embeddings and results from the previous experiment.

## Automation Logic

The `./main_process.py` script (usage [here](#usage)) automates the following steps:

1. [Preprocessing feature files](#step-1-preprocessing)
2. [Building and evaluating node embeddings](#step-2-building-and-evaluating-node-embedding)
3. [Exporiting results and Saving T-SNE plots](#step-3-export-results)

### Step 1. Preprocessing

+ The `./generate_fasttext_word_feature.py` script queries the pre-trained Facebook fastText word embedding model as word features.
+ The `./generate_identity_word_feature.py` script generates one-hot encoding as word features.
+ Other customized word features files can be store in the `./data/<dataset>/feature_matrices/<attackType>_<attackPlatform>/` folder.
+ The `./preprocess_data.py` script parse word feature CSVs and generate corresponding `.features` files under the `./data/<dataset>/` folder.

### Step 2. Building and Evaluating Node Embedding

The `./evaluation.py` script builds node embeddings for the selected dataset and evaluate the quality of generated embeddings.

### Step 3. Exporting Results

The `./export_clustering_result.py` and `./plot_tsne.py` scripts exports experiment results to the following folders:

+ Dataset configuration: `./data/<dataset>/<attackType>_<attackPlatform>_config.json`
+ Embeddings: `./embeddings/<dataset>/<feature file>/<GE model>.nv`  

    ```text
    #nodes #dim
    n0 e01 e02 e03 ... e0n
    n1 e11 e12 e13 ... e1n
    n2 ...
    .  ...
    ```

+ Runtime data: `./results/<dataset>/<attackType>_<attackPlatform>/<feature file>/experiment.cache`
+ Evaluation results:
  + Mean Average Precision: `./results/<dataset>/<attackType>_<attackPlatform>/<feature file>/MAP.xlsx`
  + KMeans results (cluster labels): `./results/<dataset>/<attackType>_<attackPlatform>/<feature file>/KMeans_labels.xlsx`
  + KMeans results (performance): `./results/<dataset>/<attackType>_<attackPlatform>/<feature file>/KMeans_performance.xlsx`

## Usage

```text
usage: gva.py [-h] --org ORG --dataset {user,repo} --n_clusters N_CLUSTERS
              [--have_features HAVE_FEATURES]
              [--weighted_graph WEIGHTED_GRAPH] [--models MODELS]
              [--commit_edgelist COMMIT_EDGELIST] [--step {P,B,E,T,A}]

Automated GVA Processor.

optional arguments:
  -h, --help                        show this help message and exit
  --dataset         {title only,title and description}     Select 'title only' or 'title and description' dataset.
  --have_features   HAVE_FEATURES   Whether the network has nodal features, default=True.
  --weighted_graph  WEIGHTED_GRAPH  Whether the edges are weighted, default=False.
  --models          MODELS          Comma delimited model names (e.g., TADW,GCAE,GATE),
                                    default=TADW,GCAE,GATE.
  --step            {P,B,E,A}     Perform a particular step ([P]reprocess, [B]uild embedding,
                                    [E]xport results) or [A]ll steps), default=A.
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
   ```
