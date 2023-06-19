import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import GroupKFold

def get_inner_files(path, extension):
    files = {
        os.path.relpath(os.path.join(root, fname), start=path)
        for root, _, files in os.walk(path)
        for fname in files
        if os.path.splitext(fname)[1].lower() == f".{extension}"
    }
    return sorted(files)


class PreProcessDataset(Dataset):
    def __init__(self, val_idx, image_path, classes, is_train=True, transforms=None):
        remove_list = ['ID325/image1664846270124.npz','ID058/image1661392103627.npz','ID089/image1661821711879.npz',
                      'ID469/image1666659964131.npz','ID363/image1664935962797.npz',
    
                       ]
        _filenames = np.load(os.path.join(image_path,'train.npy'))
        remove_idx = np.where(np.isin(_filenames,remove_list))
        _filenames = np.delete(_filenames,remove_idx)
        groups = [os.path.dirname(fname) for fname in _filenames]
        ys = [0 for _ in _filenames]
        gkf = GroupKFold(n_splits=5)
        filenames=  []
        for i, (_, y) in enumerate(gkf.split(_filenames, ys, groups)):
            if is_train == True and i != val_idx:
                filenames += list(_filenames[y])

            elif is_train == False and i == val_idx:
                filenames = list(_filenames[y])
                break
        self.image_path = image_path
        self.val_idx = val_idx
        self.classes = classes
        self.is_train = is_train
        self.transforms = transforms
        # self.image_files = np.load(os.path.join(image_path,'train.npy'))
        self.image_files = filenames
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_path,image_name)
        data = np.load(image_path)
        image = data['img']
        label = data['label']
        
        image = image/255.
        
        if self.transforms:
            inputs = (
                {"image": image, "mask": label} if self.is_train else {"image": image}
            )
            result = self.transforms(**inputs)

            image = result["image"]
            label = result["mask"] if self.is_train else label
        image = image.transpose(2, 0, 1)  # make channel first
        label = label.transpose(2, 0, 1)

        # image = torch.from_numpy(image).float()
        # label = torch.from_numpy(label).float()

        return image,label#,image_name
    
if __name__ == '__main__':
    classes = [ "finger-1","finger-2","finger-3","finger-4","finger-5","finger-6","finger-7","finger-8",
        "finger-9","finger-10","finger-11","finger-12","finger-13","finger-14","finger-15","finger-16","finger-17",
        "finger-18","finger-19","Trapezium","Trapezoid","Capitate","Hamate","Scaphoid","Lunate","Triquetrum","Pisiform","Radius","Ulna"]
    dataset = PreProcessDataset(0,'/opt/ml/level2_cv_semanticsegmentation-cv-01/newdataset/train',classes,True,None)
    # print(dataset.image_files)
    print(len(dataset))
    # img,label = dataset[0]
    # print(img.shape,label.shape)
    img,label = dataset[0]




    #데이터 정확하게 저장도ㅒㅆ는지 확인용. PreprocessDataset의 return에 image_name 추가하여 사용
    # name=name[:-3]+'png'
    # print(name)
    # origin_path = '/opt/ml/data/train/DCM'

    # origin_img = cv2.imread(os.path.join(origin_path,name))/255.

    
    # class2ind = {v: i for i, v in enumerate(classes)}
    # origin_path = '/opt/ml/data/train/outputs_json'
    # import json
    # name=name[:-3]+'json'
    # label_shape = tuple(origin_img.shape[:2]) + (len(classes),)
    # origin_label = np.zeros(label_shape, dtype=np.uint8)
    # with open(os.path.join(origin_path,name), 'r') as f: 
    #     annotations = json.load(f)
    #     annotations = annotations["annotations"]
    # for ann in annotations:
    #         c = ann["label"]
    #         class_ind = class2ind[c]
    #         points = np.array(ann["points"])

    #         # polygon to mask
    #         class_label = np.zeros(origin_img.shape[:2], dtype=np.uint8)
    #         cv2.fillPoly(class_label, [points], 1)
    #         origin_label[..., class_ind] = class_label
    # print(origin_label.shape,label.shape)
    
    # print(np.array_equal(origin_label.transpose(2,0,1),label))
    