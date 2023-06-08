from dataset.XRayInferenceDataset import XRayInferenceDataset
from dataset.XRayTrainDataset import XRayTrainDataset
import os
from torch.utils.data import Dataset,DataLoader
import numpy as np
import cv2
import json
import torch
from torchvision import transforms

def get_inner_files(path, extension):
    files = {
        os.path.relpath(os.path.join(root, fname), start=path)
        for root, _, files in os.walk(path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == f".{extension}"
    }
    return sorted(files)

class simpledataset(Dataset):
    def __init__(
        self, image_path, label_path, classes,
    ):
        _filenames = np.array(get_inner_files(image_path, "png"))
        _labelnames = np.array(get_inner_files(label_path, "json"))
        filenames = _filenames
        labelnames = _labelnames
        self.filenames = filenames
        self.labelnames = labelnames
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
        label_name = self.labelnames[item]
        label_path = os.path.join(self.label_path,label_name)
        # # process a label of shape (H, W, NC)
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

        return image, label,image_name
classes = [ "finger-1","finger-2","finger-3","finger-4","finger-5","finger-6","finger-7","finger-8",
            "finger-9","finger-10","finger-11","finger-12","finger-13","finger-14","finger-15","finger-16","finger-17",
            "finger-18","finger-19","Trapezium","Trapezoid","Capitate","Hamate","Scaphoid","Lunate","Triquetrum","Pisiform","Radius","Ulna"]
dataset = simpledataset(
    image_path="/opt/ml/level2_cv_semanticsegmentation-cv-01/data/train/DCM",
    label_path="/opt/ml/level2_cv_semanticsegmentation-cv-01/data/train/outputs_json",
    classes=classes,
)

save_path = '/opt/ml/level2_cv_semanticsegmentation-cv-01/newdataset/train'
for img,label,img_name in iter(dataset):

    path,name = img_name.split('/')
    if not os.path.exists(os.path.join(save_path,path)):
        os.makedirs(os.path.join(save_path,path))
    name = name.split('.')[0]#+'.npz'
    target_path = os.path.join(save_path,path,name)

    # np.save(target_path+'.npy',img)
    np.savez_compressed(target_path+'.npz',img=img,label=label)
    
file_names = dataset.filenames
for i in range(len(file_names)):
    file_names[i]=file_names[i][:-3]+'npz'
print(file_names)
np.save(save_path+'/train.npy',dataset.filenames)