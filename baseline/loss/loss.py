from torch import nn
import torch
from .boundaryloss import BoundaryLoss  


def DiceLoss(pred,target):
    y_true_f = target.flatten(2)
    y_pred_f = nn.functional.sigmoid(pred.flatten(2))
    intersection = torch.sum(y_true_f * y_pred_f,-1)

    eps = 1
    score = (2.0 * intersection + eps) / (
        torch.sum(y_true_f,-1) + torch.sum(y_pred_f,-1) + eps
    )

    score = score.mean()
    return 1-score


class FocalLoss(nn.Module):
    #Focal loss for segmentation
    #단, 데이터 특성상 각 클래스의 면적 비율이 비슷하므로 조금 다르게 구현함
    def __init__(self,gamma=0.5,alpha=None):
        self.CE = nn.BCELoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 0.01
    def forward(self,pred,target):
        if self.alpha:
            alpha=self.alpha
        else:
            #각 부분의 면적비율로 alpha 생성
            alpha = torch.sum(target,dim=(0,2,3))+self.eps
            alpha = 1/alpha
            alpha /= alpha.sum()
        #ce_loss.shape = (B,C,H,W)->(C)
        pred = nn.functional.sigmoid(pred)
        ce_loss = self.CE(pred,target)
        ce_loss = ce_loss.mean((2,3))
        weights=(1-pred).pow(self.gamma)
        weights = weights.mean((2,3))
        focal_loss = alpha*weights*ce_loss
        print(focal_loss.shape)
        focal_loss = focal_loss.mean()
        return focal_loss


class CustomLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sig = nn.Sigmoid()
        self.dice_loss = DiceLoss
        self.boundary_loss = BoundaryLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward(self,pred,target):
        dice = self.dice_loss(pred,target)
        bd = self.boundary_loss(pred,target) 
        bce = self.bce_loss(pred,target)
        loss = dice + bd+ bce
        loss /= 2
        return loss