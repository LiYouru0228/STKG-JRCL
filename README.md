# Learning Joint Relational Co-evolution in Spatial-Temporal Knowledge Graph for SMEs Supply Chain Prediction

## Overview
This repository is the implementation of the paper entitled as Learning Joint Relational Co-evolution in Spatial-Temporal Knowledge Graph for SMEs Supply Chain Prediction. ([KDD'23](https://dl.acm.org/doi/abs/10.1145/3580305.3599855))
> Youru Li, Zhenfeng Zhu, Xiaobo Guo, Linxun Chen, Zhouyin Wang, Yinmeng Wang, Bing Han, Yao Zhao: Learning Joint Relational Co-evolution in Spatial-Temporal Knowledge Graph for SMEs Supply Chain Prediction. KDD 2023: 4426-4436.

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

## Citation 
If you want to use our codes in your research, please cite:
```
@inproceedings{DBLP:conf/kdd/LiZGCWWH023,
  author       = {Youru Li and
                  Zhenfeng Zhu and
                  Xiaobo Guo and
                  Linxun Chen and
                  Zhouyin Wang and
                  Yinmeng Wang and
                  Bing Han and
                  Yao Zhao},
  title        = {Learning Joint Relational Co-evolution in Spatial-Temporal Knowledge
                  Graph for SMEs Supply Chain Prediction},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2023},
  pages        = {4426--4436},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3580305.3599855},
  doi          = {10.1145/3580305.3599855},
  timestamp    = {Fri, 18 Aug 2023 08:45:04 +0200},
  biburl       = {https://dblp.org/rec/conf/kdd/LiZGCWWH023.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## Statements
It is an open demo implementation of our principled algorithms used for academic research community only (should not be used for commercial purposes).
