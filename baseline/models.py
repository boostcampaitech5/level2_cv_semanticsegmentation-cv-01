import torch.nn as nn

from torchvision import models


def fcn_resnet50(class_len):
    model = models.segmentation.fcn_resnet50(pretrained=True)

    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, class_len, kernel_size=1)
    return model
