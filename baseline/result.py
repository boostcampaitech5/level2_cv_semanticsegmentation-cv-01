import os
import pandas as pd
import torch
import albumentations as A

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from dataset.XRayInferenceDataset import XRayInferenceDataset
from study.inference import test
from utils import set_seed


def main(args, k=1):
    seed = 21
    set_seed(seed)

    print(args)

    tf = A.Resize(256, 256)

    test_dataset = XRayInferenceDataset(
        test_path=args.test_path,
        transforms=tf,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )

    for i in range(k):
        model = torch.load(
            os.path.join(args.pretrained_dir, f"/opt/ml/level2_cv_semanticsegmentation-cv-01/pretrain/UNet_last.pth")
        )
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
        default="/opt/ml/level2_cv_semanticsegmentation-cv-01/data/test/DCM",
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
