# -*- coding: utf-8 -*-
"""
Description : An implementation of R-Drop (https://arxiv.org/pdf/2106.14448.pdf).
Authors     : lihp
CreateDate  : 2021/8/24
"""
from torch import nn
from torch.nn import functional as F


class RDrop(nn.Module):
    """
    R-Drop for classification tasks.
    Example:
        criterion = RDrop()
        logits1 = model(input)  # model: a classification model instance. input: the input data
        logits2 = model(input)
        loss = criterion(logits1, logits2, target)     # target: the target labels. len(loss_) == batch size
    Notes: The model must contains `dropout`. The model predicts twice with the same input, and outputs logits1 and logits2.
    """
    def __init__(self):
        super(RDrop, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.kld = nn.KLDivLoss(reduction='mean')

    def forward(self, logits1, logits2, target, kl_weight=1.):
        """
        Args:
            logits1: One output of the classification model.
            logits2: Another output of the classification model.
            target: The target labels.
            kl_weight: The weight for `kl_loss`.

        Returns:
            loss: Losses with the size of the batch size.
        """
        # ce_loss = (self.ce(logits1, target) + self.ce(logits2, target)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        # kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        # kl_loss = (kl_loss1 + kl_loss2) / 2
        kl_loss = kl_loss1
        # loss = ce_loss + kl_weight * kl_loss
        loss = kl_weight * kl_loss
        return loss

