'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import argparse
import json
import os
import random
from collections import OrderedDict

import numpy as np
import torch
from sklearn.model_selection import ParameterGrid
from torch_geometric.data import DataLoader
from torch_geometric.transforms import Compose

from src.data.graph_data import TextImageGraphData
from src.losses.pairwise import BPRLoss
from src.models.dcmm import DCMM
from src.tasks.ranking_tester import RankingTester
from src.tasks.trainer import GraphRankingTrainer
from src.utils.pytorch_ir_metrics import map_, p_k, ndcg
from src.utils.transforms import NormalizeNodes, TopKTransformer

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class CMModelFactory(object):
    @classmethod
    def get_model(self, model_config):
        str_model_name = model_config["model"]
        if str_model_name == "DCMM-cos":
            model_config["input_channels"] = [int(model_config["in_dim"])] + int(model_config["nb_layers"]) * [
                int(model_config["input_channels"])]
            del model_config["in_dim"]
            del model_config["nb_layers"]
            del model_config["model"]
            model_config["conv_type"] = "cos"
            return DCMM(**model_config)
        elif str_model_name == "DCMM-edge":
            model_config["input_channels"] = [int(model_config["in_dim"])] + int(model_config["nb_layers"]) * [
                int(model_config["input_channels"])]
            del model_config["in_dim"]
            del model_config["nb_layers"]
            del model_config["model"]
            model_config["conv_type"] = "edge"
            return DCMM(**model_config)
        else:
            raise ValueError("invalid Model Name", str_model_name)


def train_test(model, trainer_config, data_config):
    print("+++++ BEGIN TRAINING +++++")
    print(data_config)
    print("~~ number of trainable parameters ~~:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=trainer_config["lr"], weight_decay=trainer_config["l2_norm"])
    loss = BPRLoss()
    train_root = data_config["train"]
    top_k = trainer_config.get("top_k", None)
    if top_k and top_k > 0:
        graphTransform = Compose([TopKTransformer(top_k), NormalizeNodes()])
    else:
        graphTransform = NormalizeNodes()

    data_train = TextImageGraphData(root=train_root,
                                    img_features_dir=data_config["train_img_dir"],
                                    img_folder=data_config["img_featname"],
                                    transform=graphTransform,
                                    normalize="normalize",
                                    pre_compute_edges=15)  # need of a default value to create the edge index field
    train_loader = DataLoader(data_train, batch_size=trainer_config["batch_size"], shuffle=True, num_workers=5)
    # same for VALIDATION:
    val_root = data_config["val"]
    data_val = TextImageGraphData(root=val_root,
                                  img_features_dir=data_config["train_img_dir"],
                                  img_folder=data_config["img_featname"],
                                  transform=graphTransform,
                                  normalize="normalize",
                                  pre_compute_edges=15)
    validation_loader = DataLoader(data_val, batch_size=trainer_config["batch_size"] + 1, shuffle=False)
    # same for test:
    data_test = TextImageGraphData(root=data_config["test"],
                                   img_features_dir=data_config["test_img_dir"],
                                   transform=graphTransform,
                                   normalize="normalize",
                                   pre_compute_edges=15,
                                   img_folder=data_config["img_featname"]
                                   )
    test_loader = DataLoader(data_test, batch_size=trainer_config["batch_size"] + 1, shuffle=False)

    # init trainer:
    metrics = OrderedDict({"map": map_,
                           "p_k20": lambda x: p_k(x, 20),
                           "ndcg": ndcg
                           })
    trainer = GraphRankingTrainer(model, loss, optimizer, trainer_config, metrics, train_loader=train_loader,
                                  validation_loader=validation_loader, test_loader=test_loader)
    trainer.train()

    #################################################################
    # TEST
    #################################################################

    print("+++++ BEGIN TEST +++++")
    # this is not really needed
    test_config = json.load(open(os.path.join(trainer_config["checkpoint_dir"], "config.json")))
    tester = RankingTester(test_config, model)
    res = tester.test(data_test, os.path.join(trainer_config["checkpoint_dir"], "test_evaluation"))
    return res


def generic_command_line_opt(parser):
    """
    parser: an argument parser = argparse.ArgumentParser()
    """
    # mandatory
    parser.add_argument("--data", type=str)
    # mandatory
    parser.add_argument("--params", type=str)  # all parameters that will be fixed across this experiments
    # optional
    parser.add_argument("--input_channels", type=int)
    parser.add_argument("--conv_channels", type=int)
    parser.add_argument("--nb_layers", type=int, default=None)
    parser.add_argument("--nb_conv_layers", type=int, default=None)
    parser.add_argument("--visual_dim", type=int)
    # trainer options
    parser.add_argument("--loss")
    parser.add_argument("--model")
    parser.add_argument("--nb_epochs", type=int)
    parser.add_argument("--early_stopping", default=None)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--activation", default=None)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--optimizer")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--l2_norm", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--checkpoint_dir")
    # dataset options
    parser.add_argument("--top_k", type=int)
    return parser


def read_optional_config(config, args):
    """
    reads the parser option and updates the config
    """
    d = vars(args)
    for key in d:
        # not reading the data and params arguments as it will be read later anyway
        if key != "data" and key != "params":
            if d[key] is not None:
                config[key] = d[key]
    return config


def split_config_model_trainer(config):
    """
    splits a config dict into the model config and trainer config
    """
    model_config = dict(config)
    trainer_config = dict()
    for trainer_param in ["nb_epochs", "lr", "batch_size", "checkpoint_dir", "early_stopping",
                          "patience", "monitoring_metric", "loss", "l2_norm", "optimizer", "top_k"]:
        if trainer_param in config:
            trainer_config[trainer_param] = config[trainer_param]
            del model_config[trainer_param]
    return model_config, trainer_config


def load_config_json(config_file, section_name):
    with open(config_file, "r") as f:
        data = json.load(f)
    return data[section_name]


def do_standard_training():
    parser = argparse.ArgumentParser()
    generic_command_line_opt(parser)
    args = parser.parse_args()
    data_config = load_config_json(args.data, "data")
    default_config = load_config_json(args.params, "params")
    config = read_optional_config(default_config, args)
    if "cv" in data_config and data_config["cv"]:
        fold = fold_iterator(data_config, config["checkpoint_dir"])
        for fold_data_config, ckpt_dir in fold:
            model_config, trainer_config = split_config_model_trainer(config)
            trainer_config["checkpoint_dir"] = ckpt_dir
            model = CMModelFactory.get_model(model_config)
            train_test(model, trainer_config, fold_data_config)
    else:
        model_config, trainer_config = split_config_model_trainer(config)
        model = CMModelFactory.get_model(model_config)
        train_test(model, trainer_config, data_config)


def do_grid_training():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--grid", type=str)  # all parameters that will be fixed across this experiments
    args = parser.parse_args()
    data_config = load_config_json(args.data, "data")
    grid = load_config_json(args.grid, "grid_params")
    do_grid_search(grid, data_config)


def do_grid_search(grid, data_config):
    """

    grid ex: demo_grid =
    {
    "input_channels": [16, 32],
    "conv_channels": [32],
    "nb_layers": [1],
    "batch_size": [32],
    "lr": [1e-4],
    "l2_norm": [0],
    "dropout":[0.1],
    "nb_epochs": [5],
    }
    """
    param_grid = ParameterGrid(grid)
    for pid, param in enumerate(param_grid):
        model_config, trainer_config = split_config_model_trainer(param)
        trainer_config["checkpoint_dir"] = os.path.join(param["checkpoint_dir"], "P" + str(pid))
        model = CMModelFactory.get_model(model_config)
        train_test(model, trainer_config, data_config)


def fold_iterator(data_config, checkpoint_dir):
    nb_fold = int(data_config["nb_fold"])
    for i in range(1, nb_fold + 1):
        train_dir = os.path.join(data_config["train"], str(i), "train/train")
        val_dir = os.path.join(data_config["val"], str(i), "train/val")
        test_dir = os.path.join(data_config["test"], str(i), "test")
        fold_data_config = dict(data_config)
        fold_data_config["train"] = train_dir
        fold_data_config["val"] = val_dir
        fold_data_config["test"] = test_dir
        print(fold_data_config)
        yield fold_data_config, os.path.join(checkpoint_dir, "fold-" + str(i))
