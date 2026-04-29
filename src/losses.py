import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLabelSmoothingLoss(nn.Module):
    def __init__(self, gamma=2.5, smoothing=0.1):
        super(FocalLabelSmoothingLoss, self).__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        low_conf = self.smoothing / (pred.size(-1) - 1)
        one_hot = torch.full_like(pred, low_conf)
        one_hot.scatter_(1, target.unsqueeze(1), self.confidence)
        
        log_prob = F.log_softmax(pred, dim=-1)
        ce_loss = -(one_hot * log_prob).sum(dim=-1)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss