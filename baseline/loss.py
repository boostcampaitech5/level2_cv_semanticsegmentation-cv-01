import torch
import torch.nn.functional as F
import torch.nn as nn

def focal_loss(pred, gt, alpha=.25, gamma=2):
    pred = F.sigmoid(pred)
    pred = pred.view(-1)
    gt = gt.view(-1)
    BCE = F.binary_cross_entropy(pred, gt, reduction='mean')
    BCE_EXP = torch.exp(-BCE)
    loss = alpha * (1-BCE_EXP) ** gamma * BCE
    return loss

def dice_loss(pred, gt, smooth=1):
    pred = pred.contiguous()
    gt = gt.contiguous()
    intersection = (pred * gt).sum(dim=2).sum(dim=2)
    loss = (1-((2.*intersection+smooth)/
               (pred.sum(dim=2).sum(dim=2) + gt.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def IoU_loss(pred, gt, smooth=1):
    pred = F.sigmoid(pred)
    pred = pred.view(-1)
    gt = gt.view(-1)
    intersection = (pred * gt).sum()
    total = (pred + gt).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU

# TODO Tversky Loss
class TverskyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, smooth=1, alpha=0.5, beta=0.5):
        pred = pred.view(-1)
        gt = gt.view(-1)

        TP = (pred * gt).sum()
        FP = ((1-gt) * pred).sum()
        FN = (gt * (1-pred)).sum()

        loss = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        return 1 - loss

def weighted_sum_loss(pred, gt, weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, gt)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, gt, 0.2)
    loss = bce * weight + dice * (1-weight)

    return loss