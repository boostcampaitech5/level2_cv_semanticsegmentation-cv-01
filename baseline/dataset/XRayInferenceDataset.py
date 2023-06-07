import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset


def get_inner_files(path, extension):
    files = {
        os.path.relpath(os.path.join(root, fname), start=path)
        for root, _, files in os.walk(path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == f".{extension}"
    }
    return sorted(files)


class XRayInferenceDataset(Dataset):
    def __init__(self, test_path, transforms=None):
        _filenames = get_inner_files(path=test_path, extension="png")
        _filenames = np.array(sorted(_filenames))

        self.filenames = _filenames
        self.transforms = transforms
        self.test_path = test_path

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):
        image_name = self.filenames[item]
        image_path = os.path.join(self.test_path, image_name)

        image = cv2.imread(image_path)
        image = image / 255.0

        if self.transforms is not None:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result["image"]

        # to tenser will be done later
        image = image.transpose(2, 0, 1)  # make channel first

        image = torch.from_numpy(image).float()

        return image, image_name
