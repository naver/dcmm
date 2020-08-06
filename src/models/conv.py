'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops


class CMCosConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr="add", flow="target_to_source"):
        super().__init__(aggr=aggr, flow=flow)
        self.W_nodes = torch.nn.Linear(in_channels, out_channels)

    def forward(self, batch_graph):
        x_t = batch_graph.x  # text part (== nodes)
        x_v = batch_graph.x_v  # visual part (for edges)
        edge_index = batch_graph.edge_index
        edge_index, _ = remove_self_loops(edge_index)
        h = self.W_nodes(x_t)
        return self.propagate(edge_index=edge_index, x_v=x_v, h=h, num_nodes=x_t.size(0))

    def message(self, x_v_i, x_v_j, h_j):
        # see: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html for the message passing logic
        alpha_ij = torch.sum(x_v_i * x_v_j, dim=1, keepdim=True)
        # image features are L2 normalized during pre-processing, so alpha_ij == cosine sim
        return alpha_ij * h_j


class CMEdgeConv(MessagePassing):

    def __init__(self, in_channels, out_channels, visual_dim, aggr="add", flow="target_to_source"):
        super().__init__(aggr=aggr, flow=flow)
        self.W_nodes = torch.nn.Linear(in_channels, out_channels)
        self.a = Parameter(torch.Tensor(visual_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.a)

    def forward(self, batch_graph):
        x_t = batch_graph.x
        x_v = batch_graph.x_v
        edge_index = batch_graph.edge_index
        edge_index, _ = remove_self_loops(edge_index)
        h = self.W_nodes(x_t)
        return self.propagate(edge_index=edge_index, x_v=x_v, h=h, num_nodes=x_t.size(0))

    def message(self, x_v_i, x_v_j, h_j):
        alpha_ij = torch.sum(x_v_i * x_v_j * self.a, dim=1, keepdim=True)
        # a is used (and learned) to weight each dim of visual features when computing cosine sim
        return alpha_ij * h_j
