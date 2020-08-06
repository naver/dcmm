'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.nn import knn_graph


class TextImageGraphData(Dataset):
    """
    dataset class for bi-modal text-image graph objects
    * text features (i.e. LTR features) on nodes
    * image features on edges
    see pyg doc for more details on custom datasets: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
    """

    def __init__(self, root, img_features_dir, img_folder, normalize, pre_compute_edges=None, transform=None,
                 pre_transform=None):
        self.img_features_dir = img_features_dir
        self.pre_compute_edges = pre_compute_edges
        self.img_folder = img_folder
        self.normalize = True if normalize == "normalize" else False
        data = pd.read_csv(os.path.join(root, "raw/interactions.csv"), sep=",", header=0)
        self.list_features_names = sorted([column for column in data.columns if column.endswith("_feature")])
        # all the LTR features (== node features) in raw/interactions.csv should have a name ending with "_feature"
        # At init, we compute the expected names for processed files; then, if they do not exist, the process method
        # will be called
        self.q_groups = data.groupby(["query_id"])
        self.files = []
        for j in range(len(self.q_groups)):
            self.files.append("graph_{}.pt".format(j))
        super(TextImageGraphData, self).__init__(root, transform=transform, pre_transform=pre_transform)

    @property
    def raw_file_names(self):
        return ["interactions.csv"]

    @property
    def processed_file_names(self):
        return self.files

    def len(self):
        return len(self.processed_file_names)

    def download(self):
        pass

    def process(self):
        """
        process the data: from an interactions file (LETOR-like data) to graphs (one graph == one file per query)
        this is done only once
        """
        # mappings between initial q/img ids and corresponding ids in built graphs:
        mapping_query_ids = {}
        mapping_img_ids = {}
        for i, (id_, impression) in enumerate(self.q_groups):
            q_id = int(impression["query_id"].iloc[0])
            mapping_query_ids[i] = str(q_id)
            deep_features = np.load(
                os.path.join(self.img_features_dir, self.img_folder, "{}_PreLogits.npy".format(q_id)))
            imlist = []  # this holds a mapping between img id and row id in the img features matrix
            with open(os.path.join(self.img_features_dir, "imlist", "{}_imlist.txt".format(q_id))) as handler:
                for line in handler:
                    imlist.append(int(line.rstrip()))
            img_ids = [imlist.index(id_) for id_ in impression.img_id]  # == rows in deep_features
            for id_ in impression.img_id:
                if id_ not in mapping_img_ids:
                    mapping_img_ids[id_] = len(mapping_img_ids)
            img_ids_int = [mapping_img_ids[id_] for id_ in impression.img_id]
            deep_features = deep_features[img_ids]
            if self.normalize:
                deep_features = deep_features / (np.linalg.norm(deep_features, axis=1, keepdims=True) + 10e-7)
            graph = self.build_data_object(deep_features, impression)  # => builds the pyg data object
            graph.img_ids = torch.tensor(img_ids_int)
            graph.query_id = torch.tensor([i])
            if self.pre_compute_edges:
                graph.edge_index = self.compute_edges(graph)
            torch.save(graph, os.path.join(self.processed_dir, "graph_{}.pt".format(i)))
        with open(os.path.join(self.root, "mapping_query_ids.json"), "w") as handler:
            json.dump(mapping_query_ids, handler)
        with open(os.path.join(self.root, "mapping_img_ids.json"), "w") as handler:
            json.dump(mapping_img_ids, handler)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, "graph_{}.pt".format(idx)))

    def build_data_object(self, deep_features, impression):
        x = impression[self.list_features_names].values
        data = Data(x=torch.from_numpy(x).float(),
                    y=torch.tensor(list(impression.rel)).float())
        data.x_v = torch.from_numpy(deep_features).float()
        return data

    def compute_edges(self, graph):
        # for details on flow argument see:
        # 1. https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/knn_graph.html?highlight=knn_graph
        # 2. https://github.com/rusty1s/pytorch_geometric/issues/126
        # PS (in our experiments we assumed that visual descriptors were l2 normalized)
        return knn_graph(graph.x_v, self.pre_compute_edges, loop=True, flow="target_to_source")
