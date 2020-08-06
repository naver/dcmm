## Differentiable Cross Modal Model (DCMM)

This repository contains the code implementing the model introduced
in [Learning to Rank Images with Cross-Modal Graph Convolutions, ECIR'20](https://link.springer.com/chapter/10.1007/978-3-030-45439-5_39), by Thibault Formal, Stéphane Clinchant, Jean-Michel Renders, Sooyeol Lee and Geun Hee Cho.

In this paper, we proposed a reformulation of unsupervised cross-modal pseudo relevance feedback mechanisms for image search, as a differentiable architecture relying on graph convolutions.
Indeed, we can see the problem as a supervised representation learning task on graphs, and design graph convolutions operating jointly over text and image features (namely cross-modal graph convolutions). The proposed architecture directly learns how to combine image and text features for the ranking task, while taking into account the context given by all the other elements in the set of images to be (re-)ranked.

#### Requirements

The code builds on [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric), a library designed to train graph neural networks. 

Minimal requirements: 
- torch==1.5.1
- torch-geometric==1.5.0
- pandas==1.0.2
- tensorboard==2.2.2
- pytrec-eval==0.4
- tabulate==0.8.7

#### To get started

First, we recommend having a look at the PyTorch geometric library. Second, we provide a small dataset of preprocessed data from a [Mediaval'17](http://www.multimediaeval.org/mediaeval2017/) challenge. Please have a look at the `README` in the [exp](exp/README.md) folder to download the data and start training your own models !

```
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
```
