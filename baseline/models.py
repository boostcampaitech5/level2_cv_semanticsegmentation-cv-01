import torch.nn as nn

from torchvision import models


def fcn_resnet50(class_len):
    model = models.segmentation.fcn_resnet50(pretrained=True)

    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, class_len, kernel_size=1)
    return model

def deeplabv3(class_len):
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, class_len)
    return model