from __future__ import print_function, absolute_import

import numpy as np
import torch
import torch.nn.functional as F


def margin_linear(scores_1, scores_2, better_labels, margin=0.1, margin_l2=False):
    better_pm = 2 * scores_1.new(better_labels) - 1
    zero_ = scores_1.new([0.])
    margin = scores_1.new([margin])
    if not margin_l2:
        pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (scores_2 - scores_1)))
    else:
        pair_loss = torch.mean(
            torch.max(zero_, margin - better_pm * (scores_2 - scores_1)) ** 2 / np.maximum(1., margin))
    return pair_loss


def binary_cross_entropy(scores_1, scores_2, better_labels):
    compare_score = torch.sigmoid(scores_2 - scores_1)
    pair_loss = F.binary_cross_entropy(compare_score, compare_score.new(better_labels))
    return pair_loss

