# Learning Joint Relational Co-evolution in Spatial-Temporal Knowledge Graph for SMEs Supply Chain Prediction

## Overview
This repository is the implementation of the paper entitled as Learning Joint Relational Co-evolution in Spatial-Temporal Knowledge Graph for SMEs Supply Chain Prediction.

![](https://github.com/LiYouru0228/STKG-JRCL/blob/main/framework.png?raw=true)

Graphical illustration of learning joint relational co-evolution in spatial-temporal knowledge graph for SMEs supply chain prediction. It is mainly composed of three modules: (a) Multi-view Relation Sequences Mining (MvR); (b) Relational Co-evolution Learning (CoEvo); (c) Multiple Random Subspaces (MRS).

## Required packages:
The code has been tested by running a demo pipline under Python 3.9.7, and some main following packages installed and their version are:
- PyTorch == 1.10.1
- numpy == 1.21.2
- networkx == 2.8.2
- gensim == 4.1.2
- scikit-learn == 1.0.1

## Running the code
Firstly, you can run "load_data.py" to finish the data preprocessing and this command can save the preprocessed data into some pickel files. Therefore, you only need to run it the first time.

```
$ python ./src/load_data.py
```
Then, you can start to train the model and evaluate the performance by run:
```
$ python ./src/train.py
```

## Statements
This open demo implementation is used for academic research only.
