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
            T.Resize((224,224)),
            # normalize to imagenet mean and std values
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

class DataLoader(data.Dataset):
    def __init__(self, video_folder, img_size, frame_len, video_names):#resize_height, resize_width, time_step=4, num_pred=1
        self.dir = video_folder
        self.AUs_root = 'E:/dataset/ZDL/DOLOS/AUs/4f/'
        self.img_size = img_size
        self.frame_len = frame_len
        # self.useful = [6,1,30,28]
        self.kps = [5]
        self.sproot = 'E:/dataset/ZDL/DOLOS/limb/8f/'
        self.Max_len = 35

        self.video_names = video_names
        for video_name in video_names:
            video_path = os.path.join(self.dir, video_name)
            img_names = [i for i in os.listdir(video_path) if i.split('.')[-1] == 'jpg']
            vlen = len(img_names)
            if vlen<60:
              print(video_name,vlen)
        

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
            print("error label",video_name)
        video_path = os.path.join(self.dir, video_name)
        img_names = [i for i in os.listdir(video_path) if i.split('.')[-1] == 'jpg']
        # tgt_au = [14, 11, 3, 7, 6, 15]
        tgt_au = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        AUs_name = video_name + '.npy'
        AUs_path = os.path.join(self.AUs_root, AUs_name)
        raw_AUs = np.load(AUs_path, allow_pickle=True)
        AUs = []
        for n in tgt_au:
            au_events = raw_AUs[n]
            if au_events.size == 0 or au_events.shape[0] == 0:
                AUs.append(np.zeros((35, 4), dtype=np.float32))
            else:
                au_events = np.asarray(au_events, dtype=np.float32)
                if au_events.shape[0] < 35:
                    pad = np.zeros((35 - au_events.shape[0], 4), dtype=np.float32)
                    au_events = np.vstack([au_events, pad])
                elif au_events.shape[0] > 35:
                    au_events = au_events[:35]
                AUs.append(au_events.copy())

        AUs = np.stack(AUs, axis=0).astype(np.float32)

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
        # keypoint_order = ['Nose', 'left_eye', 'right_eye', 'left_ear', 'right_shoulder', 'Neck']
        keypoint_order = ['Nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'Head', 'Neck']
        sp_dict = np.load(sp_path, allow_pickle=True).item()  # .item() to get dict
        # For each keypoint, get its trajectory, pad to Max_len
        all_kp_features = []
        for kp in keypoint_order:
            if sp_dict[kp].size > 0:
                traj = sp_dict[kp].astype(np.float32)  # shape: (T, D)
                # Pad/truncate to Max_len
                if traj.shape[0] < 35:
                    pad = np.zeros((35 - traj.shape[0], traj.shape[1]), dtype=np.float32)
                    traj = np.vstack([traj, pad])
                else:
                    traj = traj[:35]
            else:
                # If missing, use zero padding
                traj = np.zeros((35, 10), dtype=np.float32)  # assuming 10-dim features

            all_kp_features.append(traj.copy())
        sp_tensor = np.stack(all_kp_features, axis=0).astype(np.float32)

        return img_samples, AUs, sp_tensor, label, video_name

    def __len__(self):
        return len(self.video_names)
