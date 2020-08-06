'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''

import torch
from torch_geometric.nn import knn_graph

"""
contains transform classes for graph data 
"""


class NormalizeEmbeddings():
    def __call__(self, graph):
        graph.x_v = graph.x_v / (torch.norm(graph.x_v, dim=1, keepdim=True) + 1e-7)
        return graph


class NormalizeNodes():
    def __call__(self, graph):
        graph.x = (graph.x - torch.mean(graph.x, dim=0)) / (
                torch.std(graph.x, unbiased=True, dim=0, keepdim=True) + 10e-7)
        return graph


class TopKTransformer():
    """
    transformer used to (re-)build the k-nn (visual) adjacency matrix of a graph object
    """

    def __init__(self, top_k=10):
        self.top_k = top_k

    def __call__(self, graph):
        edge_index = knn_graph(graph.x_v, self.top_k, loop=True, flow="target_to_source")
        graph.edge_index = edge_index
        return graph
