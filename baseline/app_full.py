import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import albumentations as A

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from dataset.XRayTrainDataset import XRayTrainDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss.loss import DiceBCE
from models import *
from hrnet.hrnet_48w_ocr import hrnet_48w_ocr
from study.train import *
from utils import *


def main(args):
    torch.cuda.empty_cache()

    seed = 21
    set_seed(seed)
    print(args)

    wandb.init(project="segmentation", name=args.model_name)

    tf = A.Compose(
        [
            A.Resize(args.resize, args.resize),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(alpha=50, sigma=10, alpha_affine=4, p=0.2),
            A.Normalize(
                mean=(0.121, 0.121, 0.121),
                std=(0.1641, 0.1641, 0.1641),
                max_pixel_value=1,
            ),
        ]
    )
    train_dataset = XRayTrainDataset(
        val_idx=-1,
        image_path=args.image_path,
        label_path=args.label_path,
        classes=args.classes,
        is_train=True,
        transforms=tf,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )

    model = hrnet_48w_ocr()

    # Loss function 정의
    criterion = DiceBCE()

    # Optimizer 정의
    optimizer = optim.AdamW(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=1e-4)
    full_train(
        model,
        args,
        train_loader,
        criterion,
        optimizer,
        scheduler=scheduler,
    )


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
        default="/opt/ml/level2_cv_semanticsegmentation-cv-01/data/train/DCM",
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
        default="fcn_res50",
    )
    parser.add_argument("--num_epoch", type=int, default=80)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=4)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
