import numpy as np
from collections import OrderedDict
from collections import Counter
import os
import glob
import cv2
import torchaudio
import torch
import torch.utils.data as data
import torchvision.transforms as T
import json
rng = np.random.RandomState(2020)
from PIL import Image
import cv2
def np_load_frame(filename, annotation, img_size):
    image_decoded = cv2.imread(filename)
    res = []
    #image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    for i  in range(5):
        x1, y1 = annotation[i][0][0], annotation[i][0][1]
        x2, y2 = annotation[i][1][0], annotation[i][1][1]
        #print(x1 - x2, y1 - y2)
        cropped_image = image_decoded[y1:y2, x1:x2]
        #print( filename , i, " ",cropped_image.shape)
        if(cropped_image.shape[0]>30 and cropped_image.shape[1]>30):
            image_resized = cv2.resize(cropped_image, ((img_size, img_size)))
        else:
            image_resized = np.zeros((img_size,img_size,3),dtype=np.float32)
        image_resized = image_resized.astype(dtype=np.float32)
        image_resized = (image_resized / 127.5) - 1.0
        res.append(image_resized)
    return res


class DataLoader(data.Dataset):
    def __init__(self, video_folder,annotations_folder, img_size,frame_len):#resize_height, resize_width, time_step=4, num_pred=1
        self.dir = video_folder
        self.annotations_folder = annotations_folder
        self.img_size = img_size
        self.frame_len = frame_len
        self.transforms = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            #T.Resize(frame_size),
            # normalize to imagenet mean and std values
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.videos = OrderedDict()
        #self._resize_height = resize_height
        #self._resize_width = resize_width
        #self._time_step = time_step
        #self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            self.videos[video_name]['frame'] = sorted(glob.glob(os.path.join(video, '*.jpg')),key=lambda x: int(os.path.basename(x).split('.')[0]))
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            vid_len = len(self.videos[video_name]['frame'])
            i = 0
            while(i + self.frame_len < vid_len):
                frames.append(self.videos[video_name]['frame'][i])
                i = i + self.frame_len
        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        if "lie" in video_name:
            label = 1
        elif "truth" in video_name:
            label = 0
        else:
            print("error label")
        idx = self.videos[video_name]['frame'].index(self.samples[index])
        annotation_file = os.path.join(self.annotations_folder,video_name+".json")
        with open(annotation_file) as json_file:
            #  load alphapose data from json file
            pic = json.load(json_file)
            #annotation = pic[frame_name]
        batch = []

        for i in range(self.frame_len):
            frame = self.videos[video_name]['frame'][i+idx]
            frame_name = frame.split('/')[-1]
            #print(frame)
            annotation  = pic[frame_name]
            batch.append(np_load_frame(frame,annotation,self.img_size))

        #image_decoded = cv2.imread(self.videos[video_name]['frame'][index])
        #image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
        #image_decoded = image_decoded.astype(dtype=np.float32)
        #batch.append(image_decoded)
        batch = np.asarray(batch)
        frame_len,blocks,height,width,channels = batch.shape
        json_file.close()

        return batch.reshape(channels,frame_len,blocks,height,width),label#batch x frame_len x blocks x width x height x channels

    def __len__(self):
        return len(self.samples)

