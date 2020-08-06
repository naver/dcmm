'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import numpy as np
import torch

"""
some IR metrics in pytorch
* we assume that relevance judgments are binary
* we do not use masking here: but if we assume masked scores are the lowest, and labels for
# masks are set to 0, it is fine 
"""


def rank_labels(scores, labels):
    """
    returns ranked labels based on scores
    scores and labels have shape (bs, list_size)
    """
    idx = torch.argsort(scores, dim=1, descending=True)  # => gives the ranked ids for each list of score
    ranked_labels = torch.gather(labels, dim=1, index=idx)
    return ranked_labels


def ndcg(labels):
    """
    computes ndcg for a batch of ranked binary labels
    """
    perfect_labels, _ = torch.sort(labels, dim=1, descending=True)
    denom = labels.new(np.log2(1 + np.arange(1, labels.size(1) + 1)))
    ndcg_max = torch.sum(perfect_labels / denom, dim=1)
    return torch.mean(torch.sum(labels / denom, dim=1) / (ndcg_max + 10e-8))


def map_(labels):
    """
    computes map for a batch of ranked binary labels
    """
    nb_relevant = torch.sum(labels, dim=1)
    denom = labels.new(np.arange(1, labels.size(1) + 1))
    return torch.mean(torch.sum((torch.cumsum(labels, dim=1) * labels) / denom, dim=1) / (nb_relevant + 10e-8))


def p_k(labels, k=5):
    """
    computes precision at rank k for a batch of ranked binary labels
    """
    k_ = min(k, labels.size(1))
    subset = labels[:, :k_]  # we only keep the k_ first columns
    return torch.mean(torch.mean(subset, dim=1))
