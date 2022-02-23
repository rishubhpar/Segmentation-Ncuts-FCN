import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os


class pascal_dataset(Dataset):
    def __init__(self, root_folder, set_type, img_transforms, seg_transforms):
        self.img_tforms = img_transforms
        self.seg_tforms = seg_transforms
        self.img_names = []
        self.root_folder = root_folder


        img_root_path = self.root_folder + '/JPEGImages/'
        seg_root_path = self.root_folder + '/SegmentationClass/'

        
        set_path = root_folder + '/ImageSets/Segmentation/' + set_type + '.txt'
        with open(set_path, 'r') as fl:
            for name in fl:
                name_edited = name[:-1]
                # name_edited = name[:-7] + '0' + name[-7:-2]
                img_path = os.path.join(img_root_path, name_edited + '.jpg')
                seg_path = os.path.join(seg_root_path, name_edited + '.png')

                # Filtering the images which are really present in the dataset 
                # print("paths:", img_path, seg_path)
                if (os.path.exists(img_path) and os.path.exists(seg_path)):
                    self.img_names.append(name_edited)

        print("Number of images in set: ", set_type, " : ", len(self.img_names))

    def reindex_seg(self, seg_img):
        idxs_none = seg_img == 255
        
        seg_rdx = seg_img.clone()
        seg_rdx[idxs_none] = 0
        return seg_rdx

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # temp_path = self.root_folder + '/JPEGImages/'
        # print("patth exists: ", os.path.exists(temp_path))

        img_path = self.root_folder + '/JPEGImages/' + img_name + '.jpg'
        seg_path = self.root_folder + '/SegmentationClass/' + img_name + '.png'

        img_orig = Image.open(img_path)
        seg_orig = Image.open(seg_path) 

        img_tform = self.img_tforms(img_orig)
        seg_tform = self.seg_tforms(seg_orig)[0,:,:] * 255 # Taking the first and only channel
        seg_reindex = self.reindex_seg(seg_tform).type(torch.LongTensor)

        # print("unique in segm: ", seg_reindex.unique()) 
        data = {'X': img_tform,
                'Y': seg_reindex}

        return data

    def __len__(self):
        return len(self.img_names)



def run_main():
    img_tforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop((480, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    seg_tforms = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop((480, 320)),
        transforms.ToTensor()])


    root_folder = './data_pascal/VOCdevkit/VOC2012'
    ds = pascal_dataset(root_folder, 'val', img_tforms, seg_tforms)

    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)

    for id, data in enumerate(dataloader):
        x = data['X']
        y = data['Y']

        print("X : ", x.shape, " Y : ", y.shape) 


if __name__ == "__main__":
    run_main()