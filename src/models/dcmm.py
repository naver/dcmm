'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import torch
from torch.nn import Linear, ReLU

from src.models.conv import CMCosConv, CMEdgeConv


class DCMM(torch.nn.Module):
    """
    Differentiable Cross Modal Model
    """

    def __init__(self,
                 input_channels=2,
                 conv_channels=2,
                 nb_conv_layers=1,
                 dropout=0.0,
                 visual_dim=2048,
                 conv_type="cos",
                 conv_activation=None):
        super().__init__()
        self.init_dict = {"input_channels": input_channels,
                          "conv_channels": conv_channels,
                          "dropout": dropout,
                          "conv_activation": conv_activation,
                          "nb_conv_layers": nb_conv_layers,
                          "visual_dim": visual_dim,
                          "conv_type": conv_type
                          }
        if conv_type == "cos":
            convClass = CMCosConv
        elif conv_type == "edge":
            convClass = lambda in_channels, out_channels: CMEdgeConv(in_channels=in_channels,
                                                                     out_channels=out_channels,
                                                                     visual_dim=visual_dim)
        else:
            raise ValueError("invalid CM convolutions argument")
        self.conv_activation = conv_activation
        self.nb_conv_layers = nb_conv_layers
        self.input_mlp = torch.nn.ModuleList()
        if len(input_channels) > 2:
            for i in range(len(input_channels) - 2):
                self.input_mlp.append(Linear(input_channels[i], input_channels[i + 1]))
                self.input_mlp.append(ReLU())
                self.input_mlp.append(torch.nn.Dropout(p=dropout))
        self.input_mlp.append(Linear(input_channels[-2], input_channels[-1]))  # final linear mapping
        self.nodes_projection = Linear(input_channels[-1], 1)  # to obtain a ranking score from node features (LTR)
        self.conv_projection = Linear(conv_channels, 1)  # to obtain a ranking score from the node embeddings after conv
        # define one or several layers of conv:
        self.conv_layers = torch.nn.ModuleList()
        self.conv_layers.append(convClass(input_channels[-1], conv_channels))
        self.dropout_conv = torch.nn.Dropout(p=dropout)
        if self.nb_conv_layers > 1:
            for _ in range(self.nb_conv_layers - 1):
                self.conv_layers.append(convClass(conv_channels, conv_channels))
        if self.conv_activation is None:
            self.activation = lambda x: x
        elif self.conv_activation == "ReLU":
            self.activation = torch.nn.ReLU()
        else:
            raise NotImplementedError()

    def forward(self, batch_graph):
        out = self.input_mlp[0](batch_graph.x)
        if len(self.input_mlp) > 1:
            for elem in self.input_mlp[1:]:
                out = elem(out)
        batch_graph.x = out
        nodes_scores = self.nodes_projection(out).squeeze()
        for conv in self.conv_layers:
            out_conv = conv(batch_graph)
            batch_graph.x = self.dropout_conv(self.activation(out_conv))
        return nodes_scores + self.conv_projection(batch_graph.x).squeeze()
