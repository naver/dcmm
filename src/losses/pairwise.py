
'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import torch
from torch_geometric.utils import to_dense_batch

from src.utils.utils import masking

"""
pairwise loss class
"""


class BasePairwiseLoss():
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __call__(self, scores, labels, batch_vec):
        """
        * the three input tensors have shape (N, ), N being the number of nodes in the batch
        * what makes possible to split values by query (i.e. graph) is the batch_vec vector, indicating which node
        belongs to which graph
        we want to compute all the pairwise contributions in the batch, dealing with:
        1. not mixing between graphs
        2. variable number of valid pairs between graphs (using masking)
        """
        ids_pos = labels == 1
        ids_neg = labels == 0
        batch_vec_pos = batch_vec[ids_pos]
        batch_vec_neg = batch_vec[ids_neg]
        pos_scores = scores[ids_pos]
        neg_scores = scores[ids_neg]
        # densify the tensors (see: https://rusty1s.github.io/pytorch_geometric/build/html/modules/utils.html?highlight=to_dense#torch_geometric.utils.to_dense_batch)
        dense_pos_scores, pos_mask = to_dense_batch(pos_scores, batch_vec_pos, fill_value=0)
        # dense_pos_scores has shape (nb_graphs, padding => max number nodes for graphs in batch)
        pos_len = torch.sum(pos_mask, dim=-1)  # shape (nb_graphs, ), actual number of nodes per graph
        dense_neg_scores, neg_mask = to_dense_batch(neg_scores, batch_vec_neg, fill_value=0)
        neg_len = torch.sum(neg_mask, dim=-1)
        max_pos_len = pos_len.max()  # == the padding value for the positive scores
        max_neg_len = neg_len.max()
        pos_mask = masking(pos_len, max_pos_len.item())
        neg_mask = masking(neg_len, max_neg_len.item())
        diff_ = dense_pos_scores.view(-1, 1, dense_pos_scores.size(1)) - dense_neg_scores.view(-1,
                                                                                               dense_neg_scores.size(1),
                                                                                               1)
        # now we use the mask and some reshaping to only extract the valid pair contributions:
        pos_mask_ = pos_mask.repeat(1, neg_mask.size(1))
        neg_mask_ = neg_mask.view(-1, neg_mask.size(1), 1).repeat(1, 1, pos_mask.size(1)).view(-1, neg_mask.size(
            1) * pos_mask.size(1))
        flattened_mask = (pos_mask_ * neg_mask_).view(-1).long()
        valid_diff_ = diff_.view(-1)[flattened_mask > 0]
        loss = self.compute_loss(valid_diff_)
        return loss

    def compute_loss(self, valid_diff_):
        raise NotImplementedError


class BPRLoss(BasePairwiseLoss):
    """
    BPR loss: compute the P(i >> j) = sigmoid(si - sj) and then do cross-entropy
    """

    def __init__(self):
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def compute_loss(self, valid_diff_):
        labels = torch.ones(valid_diff_.size(0)).to(self.device).float()  # we only have labels == 1 because we compute
        # the s(+) - s(-)
        return self.loss(valid_diff_, labels)
