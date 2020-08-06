'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import json
from collections import Counter

import pytrec_eval
import torch
from torch_geometric.utils import to_dense_batch

"""
bunch of utils functions 
"""


def agg_metrics_queries(d):
    """
    aggregates (mean) metrics over queries
    """
    final = Counter({})
    for q_id in d:
        final += Counter(d[q_id])
    n = len(d)
    return {key: value / n for key, value in final.items()}


def evaluate(qrel_file, run_file, out_path, measures=("map_cut", "map", "ndcg", "ndcg_cut", "P")):
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel_file, set(measures))
    metrics = evaluator.evaluate(run_file)
    out = agg_metrics_queries(metrics)
    with open(out_path, "w") as handler:
        json.dump(out, handler)
    return out


def masking(len_tensor, padding_size=None):
    """
    computes masking tensor from tensor of lengths
    """
    if padding_size is None:
        padding_size = torch.max(len_tensor).item()
    # compact way to compute masking: https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/5
    device = len_tensor.device
    len_shape = len_tensor.size()
    if len(len_shape) == 1:  # i.e. shape (bs,)
        indexes = torch.arange(0, padding_size).unsqueeze(0).to(device)
        mask = (indexes < len_tensor.unsqueeze(1)).float().to(device)
    elif len(len_shape) == 2:  # i.e. shape (bs, list_size)
        indexes = torch.arange(0, padding_size).unsqueeze(0).unsqueeze(0).to(device)
        mask = (indexes < len_tensor.unsqueeze(2)).float().to(device)
    else:
        raise ValueError("masking in only available for shape (bs,) and (bs, list_size)")
    return mask


def batch_sparse(scores, labels, batch):
    """
    method to convert "sparse" pyg vectors of scores and labels to dense ones
    """
    batch_scores, _ = to_dense_batch(scores, batch, fill_value=-10e8)
    batch_labels, _ = to_dense_batch(labels, batch, fill_value=0)
    return batch_scores, batch_labels
