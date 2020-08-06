'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import datetime
import json
import os
import time
from collections import Counter

import torch
from torch.utils.tensorboard import SummaryWriter

from src.tasks.early_stopping import EarlyStopping
from src.tasks.saver import Saver
from src.utils.pytorch_ir_metrics import rank_labels
from src.utils.utils import batch_sparse

"""
Trainer classes
disclaimer: strongly inspired from https://github.com/victoresque/pytorch-template
"""


class BaseTrainer:
    """base trainer class"""

    def __init__(self, model, loss, optimizer, config, metrics):
        """
        model: model object
        loss: loss object
        optimizer: optimizer object
        config: dict of config parameters
        metrics: OrderedDict of (callable) metrics, e.g. {"map": map, ...
        """
        print("initialize trainer...")
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics
        # no multi-GPUs case:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.model.train()  # put model on train mode
        self.checkpoint_dir = config["checkpoint_dir"]
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        if not os.path.exists(os.path.join(self.checkpoint_dir, "saved")):
            os.makedirs(os.path.join(self.checkpoint_dir, "saved"))
        self.nb_epochs = config["nb_epochs"]
        # setup tensorboard writer instance:
        self.writer_dir = os.path.join(config["checkpoint_dir"],
                                       "tensorboard",
                                       datetime.datetime.now().strftime('%m%d_%H%M%S'))
        self.writer = SummaryWriter(self.writer_dir)
        self.config = config
        print("trainer config:\n", self.config)
        self.config["model_init_dict"] = model.init_dict  # NOTE: each model should have a dict attribute that contains
        # everything needed to instantiate it in this fashion: model(**init_dict)
        # (handy when loading a saved model for inference)

    def train(self):
        """
        full training logic
        """
        # initialize early stopping or saver:
        if "early_stopping" in self.config:
            saver = EarlyStopping(self.config["patience"], self.config["early_stopping"])
        else:
            saver = Saver()
        t0 = time.time()
        training_res_handler = open(os.path.join(self.checkpoint_dir, "training_perf.txt"), "w")
        validation_res_handler = open(os.path.join(self.checkpoint_dir, "validation_perf.txt"), "w")
        training_res_handler.write("epoch,loss\n")
        validation_res_handler.write("epoch,loss,{}\n".format(",".join(self.metrics.keys())))
        try:
            if self.test_loader is not None:
                test_res_handler = open(os.path.join(self.checkpoint_dir, "test_perf.txt"), "w")
                test_res_handler.write("epoch,loss,{}\n".format(",".join(self.metrics.keys())))
        except AttributeError:
            print("no logging of test metrics")
        for epoch in range(1, self.nb_epochs + 1):
            print("==== BEGIN EPOCH {} ====".format(epoch))
            # == start training for one epoch ==
            self.model.train()  # => train mode
            train_loss = self.train_epoch(epoch)  # train_epoch() returns the training loss (on full training set)
            print("*train loss:{}".format(train_loss))
            self.writer.add_scalar(os.path.join(self.writer_dir, "full_train_loss"), train_loss, epoch)
            training_res_handler.write("{},{}\n".format(epoch, train_loss))
            # == start validation ==
            self.model.eval()  # => eval mode
            with torch.no_grad():
                val_loss, val_metrics = self.valid_epoch()
                # add validation loss to tensorboard:
                self.writer.add_scalar(os.path.join(self.writer_dir, "full_validation_loss"), val_loss, epoch)
                # add validation metrics to tensorboard:
                for metric in val_metrics:
                    self.writer.add_scalar(os.path.join(self.writer_dir, "full_validation_{}".format(metric)),
                                           val_metrics[metric], epoch)
                # and write these values to validation text file:
                validation_res_handler.write("{},{:5f}".format(epoch, val_loss))
                for key_ in self.metrics.keys():
                    validation_res_handler.write(",{:5f}".format(val_metrics[key_]))
                validation_res_handler.write('\n')
                # same for test (if test loader):
                try:
                    test_loss, test_metrics = self.valid_epoch(data="test")
                    # add validation loss to tensorboard:
                    self.writer.add_scalar(os.path.join(self.writer_dir, "full_test_loss"), test_loss, epoch)
                    for metric in test_metrics:
                        self.writer.add_scalar(os.path.join(self.writer_dir, "test_{}".format(metric)),
                                               test_metrics[metric], epoch)
                    test_res_handler.write("{},{:5f}".format(epoch, test_loss))
                    for key_ in self.metrics.keys():
                        test_res_handler.write(",{:5f}".format(test_metrics[key_]))
                    test_res_handler.write('\n')
                except AssertionError:
                    pass
                print("=validation-loss:{}".format(val_loss))
                for key, val in val_metrics.items():
                    print("+validation-{}:{}".format(key, val))
                if "early_stopping" in self.config:
                    if self.config["early_stopping"] == "loss":
                        saver(val_loss, self, epoch)
                    else:
                        saver(val_metrics[self.config["early_stopping"]], self, epoch)
                    if saver.stop:  # meaning we reach the early stopping criterion
                        print("== EARLY STOPPING AT EPOCH {}".format(epoch))
                        self.config["stop_iter"] = epoch
                        break
                else:
                    saver(val_metrics[self.config["monitoring_metric"]], self, epoch)
        self.writer.close()  # closing tensorboard writer
        with open(os.path.join(self.checkpoint_dir, "config.json"), "w") as handler:
            json.dump(self.config, handler)
        training_res_handler.close()
        validation_res_handler.close()
        print("======= TRAINING DONE =======")
        print("took about {} hours".format((time.time() - t0) / 3600))

    def save_checkpoint(self, epoch, val_perf, is_best=False):
        """
        """
        state = {"epoch": epoch,
                 "val_perf": val_perf,
                 "model_state_dict": self.model.state_dict(),
                 "optimizer_state_dict": self.optimizer.state_dict(),
                 "config": self.config
                 }
        if is_best:
            listdir = os.listdir(os.path.join(self.checkpoint_dir, "saved"))
            update = False
            if len(listdir) == 1:
                previous_file_name = listdir[0]
                update = True
            torch.save(state, os.path.join(self.checkpoint_dir, "saved/model_best-at_epoch_{}.tar".format(epoch)))
            if update:
                os.remove(os.path.join(self.checkpoint_dir, "saved/{}".format(previous_file_name)))  # because we
                # only want to keep the last best config

    def train_epoch(self, epoch):
        """
        epoch training logic
        """
        raise NotImplementedError

    def valid_epoch(self, **kwargs):
        """
        epoch validation logic
        """
        raise NotImplementedError


class GraphRankingTrainer(BaseTrainer):
    def __init__(self, model, loss, optimizer, config, metrics, train_loader, validation_loader, test_loader=None):
        """
        """
        super().__init__(model, loss, optimizer, config, metrics)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        if test_loader is not None:
            self.test_loader = test_loader
        else:
            self.test_loader = None

    def forward(self, batch_graph):
        """
        batch_graph: batch of graphs (torch_geometric.data.Data object)
        return: scores, labels and batch vector for the batch of graphs
        """
        for k in batch_graph.keys:
            batch_graph[k] = batch_graph[k].to(self.device)
            # => move all the tensors in batch_graph to device
        scores = self.model(batch_graph)
        # returns 1-D tensors:
        return scores, batch_graph.y, batch_graph.batch

    def train_epoch(self, epoch):
        """
        """
        # the model is already in train mode
        total_loss = 0
        for batch_id, batch_graph in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            scores, labels, batch_vec = self.forward(batch_graph)
            loss = self.loss(scores, labels, batch_vec)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def valid_epoch(self, data="validation"):
        """
        epoch validation logic
        return: validation loss and validation metrics (or test if data == "test", validation by default)
        """
        # the model is in eval mode (+ torch.no_grad())
        total_metrics = Counter({})
        total_loss = 0
        assert (data == "validation" or (data == "test" and self.test_loader is not None))
        loader = self.validation_loader if data == "validation" else self.test_loader
        for batch_id, batch_graph in enumerate(loader):
            scores, labels, batch_vec = self.forward(batch_graph)
            loss = self.loss(scores, labels, batch_vec)
            total_loss += loss.item()
            total_metrics += self.eval_metrics(scores, labels, batch_vec)
        out = {key: value / len(loader) for key, value in total_metrics.items()}
        return total_loss / len(loader), out

    def eval_metrics(self, scores, labels, batch_vec):
        """
        computes of bunch of metrics for a batch of graphs
        """
        res = {}
        # convert sparse batch encoding of scores and labels to dense tensors:
        batch_scores, batch_labels = batch_sparse(scores, labels, batch_vec)
        batch_scores = batch_scores.to(self.device)
        batch_labels = batch_labels.to(self.device)
        ranked_labels = rank_labels(batch_scores, batch_labels)
        for key, metric_fn in self.metrics.items():
            res[key] = metric_fn(ranked_labels).item()
        return Counter(res)
