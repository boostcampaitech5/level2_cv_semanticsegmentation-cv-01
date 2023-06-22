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
            # for i in range(outputs.shape[0]):
            #     for j in range(outputs.shape[1]):
            #         outputs[i][j] = crf(2048, 2048, outputs[i][j])

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{ind2class[c]}_{image_name}")

    return rles, filename_and_class


def ensemble_test(
    model1, model2, model3, model4, model5, model6, classes, data_loader, thr=0.5
):
    model1 = model1.cuda()
    model2 = model2.cuda()
    model3 = model3.cuda()
    model4 = model4.cuda()
    model5 = model5.cuda()
    model6 = model6.cuda()

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()
    model6.eval()

    ind2class = {i: v for i, v in enumerate(classes)}
    base_path = "/opt/ml/level2_cv_semanticsegmentation-cv-01/sample"

    rles = []
    filename_and_class = []
    with torch.no_grad():
        for _, (images, image_names) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            outputs = []
            images = images.cuda()
            output = model1(images)
            output = F.interpolate(output, size=(2048, 2048), mode="bilinear")
            output = torch.sigmoid(output)
            outputs.append(output)
            output = model2(images)
            output = F.interpolate(output, size=(2048, 2048), mode="bilinear")
            output = torch.sigmoid(output)
            outputs.append(output)
            output = model3(images)
            output = F.interpolate(output, size=(2048, 2048), mode="bilinear")
            output = torch.sigmoid(output)
            outputs.append(output)
            output = model4(images)
            output = F.interpolate(output, size=(2048, 2048), mode="bilinear")
            output = torch.sigmoid(output)
            outputs.append(output)
            output = model5(images)
            output = F.interpolate(output, size=(2048, 2048), mode="bilinear")
            output = torch.sigmoid(output)
            outputs.append(output)
            output = model6(images)
            output = F.interpolate(output, size=(2048, 2048), mode="bilinear")
            output = torch.sigmoid(output)
            outputs.append(output)

            outputs = torch.stack(outputs)
            outputs = torch.mean(outputs, dim=0)
            for i, output in enumerate(outputs):
                image_name = os.path.basename(image_names[i]).split(".")[0]
                np.savez_compressed(
                    f"{os.path.join(base_path, image_name)}.npz",
                    array=output.half().detach().cpu().numpy(),
                )
            outputs = (outputs > thr).detach().cpu().numpy()

            for output, image_name in zip(outputs, image_names):
                for c, segm in enumerate(output):
                    rle = encode_mask_to_rle(segm)
                    rles.append(rle)
                    filename_and_class.append(f"{ind2class[c]}_{image_name}")

    return rles, filename_and_class
