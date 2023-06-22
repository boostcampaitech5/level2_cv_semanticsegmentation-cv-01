import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm
from utils import *


def encode_mask_to_rle(mask):
    """
    mask: numpy array binary mask
    1 - mask
    0 - background
    Returns encoded run length
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def test(model, classes, data_loader, thr=0.5):
    model = model.cuda()
    model.eval()

    ind2class = {i: v for i, v in enumerate(classes)}

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for _, (images, image_names) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            images = images.cuda()
            # outputs = model(images)["out"]
            outputs = model(images)

            # restore original size
            outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu().numpy()

            # DenseCRF
            for i in range(outputs.shape[0]):
                for j in range(outputs.shape[1]):
                    outputs[i][j] = crf(2048, 2048, outputs[i][j])

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{ind2class[c]}_{image_name}")

    return rles, filename_and_class
