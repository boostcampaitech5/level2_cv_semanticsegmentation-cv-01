import os
import pandas as pd
import torch
import albumentations as A

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from dataset.XRayInferenceDataset import XRayInferenceDataset
from study.inference import *
from utils import set_seed


def main(args):
    seed = 21
    set_seed(seed)

    print(args)

    tf = A.Compose(
        [
            A.Resize(args.resize, args.resize),
            A.Normalize(
                mean=(0.121, 0.121, 0.121),
                std=(0.1641, 0.1641, 0.1641),
                max_pixel_value=1,
            ),
        ]
    )

    test_dataset = XRayInferenceDataset(
        test_path=args.test_path,
        transforms=tf,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )

    model1 = torch.load(os.path.join(args.pretrained_dir, f"{args.pretrain}0.pth"))
    model2 = torch.load(os.path.join(args.pretrained_dir, f"{args.pretrain}1.pth"))
    model3 = torch.load(os.path.join(args.pretrained_dir, f"{args.pretrain}2.pth"))
    model4 = torch.load(os.path.join(args.pretrained_dir, f"{args.pretrain}3.pth"))
    model5 = torch.load(os.path.join(args.pretrained_dir, f"{args.pretrain}4.pth"))
    model6 = torch.load(
        "/opt/ml/level2_cv_semanticsegmentation-cv-01/pretrain/hrnet_48w_ocr_1024_augSet_full.pth"
    )
    rles, filename_and_class = ensemble_test(
        model1,
        model2,
        model3,
        model4,
        model5,
        model6,
        args.classes,
        test_loader,
        args.thr,
    )
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]
    df = pd.DataFrame(
        {
            "image_name": image_name,
            "class": classes,
            "rle": rles,
        }
    )
    df.to_csv(os.path.join(args.saved_dir, f"{args.output}.csv"), index=False)
    full_df = pd.DataFrame(
        {
            "image_name": filename,
            "class": classes,
            "rle": rles,
        }
    )
    full_df.to_csv(os.path.join(args.saved_dir, f"{args.output}_full.csv"), index=False)


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
    parser.add_argument(
        "--pretrain",
        type=str,
        default="fcn_res50_cosAnn_best",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="fcn_res50_cosAnn_best",
    )
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--thr", type=float, default=0.5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
