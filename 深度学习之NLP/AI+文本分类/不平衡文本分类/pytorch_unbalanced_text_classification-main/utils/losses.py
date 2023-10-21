import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(weight, list):
            weight = torch.tensor(weight, requires_grad=False)
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        weight = self.weight.to(input.device)
        loss = F.nll_loss(logpt, target, weight, ignore_index=self.ignore_index)
        return loss
