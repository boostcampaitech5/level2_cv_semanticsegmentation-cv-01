import os
import pandas as pd
import torch
import albumentations as A

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from dataset.XRayInferenceDataset import XRayInferenceDataset
from study.inference import test
from utils import set_seed
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
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
def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)
def main(args, k=1):
    seed = 21
    set_seed(seed)

    print(args)

    tf = A.Compose([A.Resize(1024, 1024),
                    A.Normalize(mean=(0.121,0.121,0.121),std=(0.1641,0.1641,0.1641) ,max_pixel_value=1)
                    ])

    test_dataset = XRayInferenceDataset(
        test_path=args.test_path,
        transforms=tf,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )
    print(len(test_dataset))
    thr= 0.5
    ind2class = {i: v for i, v in enumerate(args.classes)}
    for i in range(k):
        model = torch.load(
            os.path.join(args.pretrained_dir, f"mmSegformer_b4_upsample_more_best1.pth")
        )
        with torch.no_grad():
            for _, (images, image_names) in tqdm(
                enumerate(test_loader), total=len(test_loader)
            ):
                images = images.cuda()
                outputs = model(images)

                # restore original size
                outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > thr).detach().cpu().numpy()
                outputs = np.where(outputs>0.5,1,0)
                rle = encode_mask_to_rle(outputs[0][0])

                decode_output = decode_rle_to_mask(rle,2048,2048)

                print(np.array_equal(outputs[0][0],decode_output))
               
        return

        rles, filename_and_class = test(model, args.classes, test_loader)
        classes, filename = zip(*[x.split("_") for x in filename_and_class])
        image_name = [os.path.basename(f) for f in filename]
        df = pd.DataFrame(
            {
                "image_name": image_name,
                "class": classes,
                "rle": rles,
            }
        )
        df.to_csv(os.path.join(args.saved_dir, "output{i}.csv"), index=False)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--classes",
        type=list,
        default=[
            "finger-1",
            "finger-2",
            "finger-3",
            "finger-4",
            "finger-5",
            "finger-6",
            "finger-7",
            "finger-8",
            "finger-9",
            "finger-10",
            "finger-11",
            "finger-12",
            "finger-13",
            "finger-14",
            "finger-15",
            "finger-16",
            "finger-17",
            "finger-18",
            "finger-19",
            "Trapezium",
            "Trapezoid",
            "Capitate",
            "Hamate",
            "Scaphoid",
            "Lunate",
            "Triquetrum",
            "Pisiform",
            "Radius",
            "Ulna",
        ],
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/opt/ml/data/test/DCM",
    )
    parser.add_argument(
        "--saved_dir",
        type=str,
        default="/opt/ml/level2_cv_semanticsegmentation-cv-01/output",
    )
    parser.add_argument(
        "--pretrained_dir",
        type=str,
        default="/opt/ml/level2_cv_semanticsegmentation-cv-01/pretrain",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
