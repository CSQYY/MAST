import numpy as np
import os
from einops import rearrange
import torch
import torch.utils.data as data
import json

rng = np.random.RandomState(2020)
from PIL import Image
import glob
import torchvision.transforms as T

transforms = T.Compose([
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32),
    T.Resize((224, 224)),
    # normalize to imagenet mean and std values
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DataLoader(data.Dataset):
    def __init__(self, video_folder, img_size, frame_len, dataset,
                 video_names):  # resize_height, resize_width, time_step=4, num_pred=1
        self.dataset = dataset
        self.dir = video_folder
        self.AUs_root = '/data01/behavior_group/d21_qiaoyy/'+self.dataset+'/AU/4f/'
        self.img_size = img_size
        self.frame_len = frame_len
        # self.useful = [6,1,30,28]
        self.kps = [5]
        self.sproot = '/data01/behavior_group/d21_qiaoyy/'+self.dataset+'/limb/8f/'
        self.Max_len = 35

        self.video_names = video_names
        for video_name in video_names:
            video_path = os.path.join(self.dir, video_name)
            img_names = [i for i in os.listdir(video_path) if i.split('.')[-1] == 'jpg']
            vlen = len(img_names)
            if vlen < 60:
                print(video_name, vlen)

    def __getitem__(self, index):
        video_name = self.video_names[index]

        if "lie" in video_name:
            label = 1
        elif "truth" in video_name:
            label = 0
        elif "deception" in video_name:
            label = 1
        elif "true" in video_name:
            label = 0
        else:
            print("error label", video_name)
        video_path = os.path.join(self.dir, video_name)
        img_names = [i for i in os.listdir(video_path) if i.split('.')[-1] == 'jpg']
        AUs_name = video_name + '.npy'
        AUs_path = os.path.join(self.AUs_root, AUs_name)
        AUs = np.load(AUs_path)

        vlen = len(img_names)
        target_frames = np.linspace(0, vlen - 1, num=self.frame_len)
        target_frames = np.around(target_frames).astype(
            int)  # certain frames may be redundant because of rounding to the nearest integer

        img_samples = []
        for i in target_frames:
            img = np.asarray(Image.open(os.path.join(video_path, img_names[i]))) / 255.0
            img_samples.append(transforms(img))

        img_samples = torch.stack(img_samples, 0).transpose(0, 1)
        img_samples.type(torch.float32)

        sp_path = self.sproot + video_name + '.npy'
        sp_lists = np.load(sp_path)

        return img_samples, AUs, sp_lists, label

    def __len__(self):
        return len(self.video_names)
