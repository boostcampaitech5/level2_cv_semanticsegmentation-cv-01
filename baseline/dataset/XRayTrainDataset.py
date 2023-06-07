import json
import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold


def get_inner_files(path, extension):
    files = {
        os.path.relpath(os.path.join(root, fname), start=path)
        for root, _, files in os.walk(path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == f".{extension}"
    }
    return sorted(files)


class XRayTrainDataset(Dataset):
    def __init__(
        self, val_idx, image_path, label_path, classes, is_train=True, transforms=None
    ):
        _filenames = np.array(get_inner_files(image_path, "png"))
        _labelnames = np.array(get_inner_files(label_path, "json"))

        # split train-valid
        # 한 폴더 안에 한 인물의 양손에 대한 `.dcm` 파일이 존재하기 때문에
        # 폴더 이름을 그룹으로 해서 GroupKFold를 수행합니다.
        # 동일 인물의 손이 train, valid에 따로 들어가는 것을 방지합니다.
        groups = [os.path.dirname(fname) for fname in _filenames]

        # dummy label
        ys = [0 for _ in _filenames]

        # 전체 데이터의 20%를 validation data로 쓰기 위해 `n_splits`를
        # 5으로 설정하여 KFold를 수행합니다.
        gkf = GroupKFold(n_splits=5)

        filenames = []
        labelnames = []
        for i, (_, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train == True and i != val_idx:
                filenames += list(_filenames[y])
                labelnames += list(_labelnames[y])

            elif is_train == False and i == val_idx:
                filenames = list(_filenames[y])
                labelnames = list(_labelnames[y])
                break

        self.filenames = filenames
        self.labelnames = labelnames
        self.is_train = is_train
        self.transforms = transforms
        self.image_path = image_path
        self.label_path = label_path
        self.classes = classes
        self.class2ind = {v: i for i, v in enumerate(classes)}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.image_path, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_path, label_name)

        # process a label of shape (H, W, NC)
        label_shape = tuple(image.shape[:2]) + (len(self.classes),)
        label = np.zeros(label_shape, dtype=np.uint8)

        # read label file
        with open(label_path, "r") as f:
            annotations = json.load(f)
        annotations = annotations["annotations"]

        # iterate each class
        for ann in annotations:
            c = ann["label"]
            class_ind = self.class2ind[c]
            points = np.array(ann["points"])

            # polygon to mask
            class_label = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(class_label, [points], 1)
            label[..., class_ind] = class_label

        if self.transforms is not None:
            inputs = (
                {"image": image, "mask": label} if self.is_train else {"image": image}
            )
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()

        return image, label
