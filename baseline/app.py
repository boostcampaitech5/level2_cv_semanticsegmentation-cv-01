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
                    A.RandomScale((0.1,0.1)) ,
                    A.PadIfNeeded(512,512),
                    A.Rotate(10),
                    A.RandomCrop(512,512),
                    A.GaussNoise(var_limit=(0,0.005),per_channel=False),
                    A.CoarseDropout(60,5,5,10),
                    A.Normalize(mean=(0.121,0.121,0.121),std=(0.1641,0.1641,0.1641) ,max_pixel_value=1)
    ])
    val_tf = A.Compose([A.Resize(args.resize, args.resize),
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
            num_workers=1,
            drop_last=False,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            drop_last=False,
        )

        model = MMSegFormer()
        
        # Loss function 정의
        criterion = nn.BCEWithLogitsLoss()

        # Optimizer 정의
        optimizer = optim.AdamW([
            {'params':model.model.backbone.parameters(), 'lr':args.lr, 'weight_decay':args.weight_decay},
            {'params':model.model.decode_head.parameters(), 'lr':args.lr*10, 'weight_decay':args.weight_decay},
        ])
        train(model, args, train_loader, valid_loader, criterion, optimizer, i, accum_step=4)


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
        default="mmSegformer_b0_augmentation",
    )
    parser.add_argument("--num_epoch", type=int, default=120)
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
