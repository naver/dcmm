'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import numpy as np


class EarlyStopping():

    def __init__(self, patience, mode):
        """early stopping object
        mode: do we do early stopping on loss or metrics ?
        """
        self.patience = patience
        self.counter = 0
        assert mode in {"loss", "map", "ndcg", "p_k20"}
        self.best = np.Inf if mode == "loss" else 0  # worst loss is Inf, worst metrics are equal to 0
        self.fn = lambda x, y: x <= y if mode == "loss" else lambda z, w: z > w
        self.stop = False
        print("init early stopping with {}, patience={}".format(mode, patience))

    def __call__(self, val_score, trainer, epoch):
        """
        early stopping call
        val_score: either validation loss or metric
        trainer: trainer object
        epoch: epoch step
        """
        if self.fn(val_score, self.best):
            # => improvement
            self.best = val_score
            self.counter = 0
            trainer.save_checkpoint(epoch, val_score, is_best=True)
        else:
            # => no improvement
            self.counter += 1
            if self.counter > self.patience:
                self.stop = True
