import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


def fcn_resnet50(class_len):
    model = models.segmentation.fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, class_len, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, class_len, kernel_size=1)
    return model


def deeplabv3_resnet50(class_len):
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, class_len, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, class_len, kernel_size=1)
    return model


def deeplabv3_resnet101(class_len):
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, class_len, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, class_len, kernel_size=1)
    return model


def deeplabv3_hrnet(class_len):
    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.backbone = timm.create_model(
        "hrnet_w18_small_v2", pretrained=False, head="None"
    )
    model.classifier[4] = nn.Conv2d(256, class_len, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, class_len, kernel_size=1)
    return model
