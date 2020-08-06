
'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
class Saver():

    def __init__(self):
        self.best = 0

    def __call__(self, val_score, trainer, epoch):
        """
        saver call
        val_score: validation metric
        trainer: trainer object
        epoch: epoch step
        """
        if val_score > self.best:
            # => improvement
            self.best = val_score
            trainer.save_checkpoint(epoch, val_score, is_best=True)
