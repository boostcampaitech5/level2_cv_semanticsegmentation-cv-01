import wandb
import torch.nn as nn
import torch.optim as optim
import albumentations as A

from argparse import ArgumentParser
from torch.utils.data import DataLoader

from dataset.XRayTrainDataset import XRayTrainDataset
from dataset.PreprocessDataset import PreProcessDataset
from dataset.PreprocessDataset_gray import PreProcessDatasetGray
from models import *
from study.train import train
from utils import set_seed
from loss.loss import *
def main(args, k=1):
    seed = 21
    set_seed(seed)
    print(args)

    wandb.init(project="segmentation", name=args.model_name)

    tf = A.Compose([A.Resize(args.resize, args.resize),
                    A.RandomScale((0.1,0.1)) ,
                    A.PadIfNeeded(args.resize,args.resize),
                    A.Rotate(10),
                    A.RandomCrop(args.resize,args.resize),
                    # A.CoarseDropout(60,5,5,10),
                    A.Normalize(mean=0.121,std=0.1641 ,max_pixel_value=1)
    ])
    val_tf = A.Compose([A.Resize(args.resize, args.resize),
                    A.Normalize(mean=0.121,std=0.1641,max_pixel_value=1)
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
            transforms=val_tf,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False,
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            drop_last=False,
        )

        model = MMSegFormer()
        
        # Loss function 정의
        # criterion = CustomLoss()
        # alpha = [0.9287, 0.9128, 0.9094, 0.9266, 0.9163, 0.9059, 0.9121, 0.9207, 0.9137,
        # 0.9055, 0.9165, 0.9223, 0.9155, 0.9107, 0.9167, 0.9340, 0.9270, 0.9110,
        # 0.9110, 0.9490, 0.9861, 0.9400, 0.9469, 0.9282, 0.9382, 0.9408, 1.0000,
        # 0.9041, 0.9083]
        alpha=0.5
        # criterion = FocalLoss(gamma=2,alpha=alpha)
        criterion = mmFocalLoss(alpha=alpha,loss_weight=3.0)
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
        default="mmSegformer_b0_upsample_gray",
    )
    parser.add_argument("--num_epoch", type=int, default=120)
    parser.add_argument("--resize", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=9e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--val_every", type=int, default=5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
