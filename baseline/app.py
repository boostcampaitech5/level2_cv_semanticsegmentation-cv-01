import wandb
import torch.nn as nn
import torch.optim as optim
import albumentations as A

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from dataset.XRayTrainDataset import XRayTrainDataset
from dataset.PreprocessDataset import PreProcessDataset
from models import *
from study.train import train
from utils import set_seed


def main(args, k=1):
    seed = 21
    set_seed(seed)
    print(args)

    wandb.init(project="segmentation", name=args.model_name)

    tf = A.Compose([A.Resize(args.resize, args.resize),
                   A.Normalize(max_pixel_value=1)
    ])
    for i in range(k):
        train_dataset = PreProcessDataset(
            val_idx=i,
            image_path=args.image_path,
            classes=args.classes,
            is_train=True,
            transforms=tf,
        )
        valid_dataset = PreProcessDataset(
            val_idx=i,
            image_path=args.image_path,
            classes=args.classes,
            is_train=False,
            transforms=tf,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            drop_last=False,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )

        model = UNet(len(args.classes))

        # Loss function 정의
        criterion = nn.BCEWithLogitsLoss()

        # Optimizer 정의
        optimizer = optim.AdamW(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        train(model, args, train_loader, valid_loader, criterion, optimizer, i, accum_step=8)


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
        "--image_path",
        type=str,
        default="/opt/ml/level2_cv_semanticsegmentation-cv-01/newdataset/train",
    )
    parser.add_argument(
        "--label_path",
        type=str,
        default="/opt/ml/level2_cv_semanticsegmentation-cv-01/data/train/outputs_json",
    )
    parser.add_argument(
        "--saved_dir",
        type=str,
        default="/opt/ml/level2_cv_semanticsegmentation-cv-01/output",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="UNet",
    )
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--resize", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=10e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
