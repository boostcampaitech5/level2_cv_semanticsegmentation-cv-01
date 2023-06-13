from torch import nn
import torch
from .boundaryloss import BoundaryLoss  
from .focalloss import MMFocalLoss

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
    #단, 데이터 특성상 이미지당 각 클래스의 면적 비율이 비슷하므로 조금 다르게 구현함
    def __init__(self,gamma=2,alpha=0.5,loss_weight=2):
        super().__init__()
        self.CE = nn.BCEWithLogitsLoss(reduction='none')
        # self.alpha = torch.FloatTensor(alpha).cuda()
        self.alpha=alpha
        self.gamma = gamma
        self.eps = 0.01
        self.loss_weight=loss_weight
    def forward(self,pred,target):
        # if self.alpha!= None:
        #     alpha=self.alpha
        # else:
        #     # 각 부분의 면적비율로 alpha 생성
        #     alpha = torch.sum(target,dim=(0,2,3))+self.eps
        #     alpha = 1/alpha
        #     alpha /= torch.sum(alpha)
            
        alpha = self.alpha*target + (1-self.alpha)*(1-target)

        #print(alpha)
        
        #ce_loss.shape = (B,C,H,W)->(C)
        ce_loss = self.CE(pred,target)
        # ce_loss = ce_loss
        weights=torch.abs(pred-nn.functional.sigmoid(pred)).pow(self.gamma)
        # weights = weights
        focal_loss = alpha*weights*ce_loss
        focal_loss = focal_loss.mean()

        return focal_loss*self.loss_weight

def mmFocalLoss(alpha=0.5,loss_weight=2.0):
    return MMFocalLoss(alpha=alpha,loss_weight=loss_weight)
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