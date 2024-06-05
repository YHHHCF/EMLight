import os
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import util
from PIL import Image
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import pickle
import imageio
imageio.plugins.freeimage.download()
import shutil


class ParameterDataset(Dataset):
    def __init__(self, dir, mode="train", small_num=128):
        assert os.path.exists(dir)

        self.pairs = []

        gt_dir = dir + 'pkl/'

        if mode == "train":
            crop_dir = dir + 'input_train/'
        elif mode == "val":
            crop_dir = dir + 'input_val/'
        elif mode == "test":
            crop_dir = dir + 'input_test/'
        elif mode == "train_small":
            # try to overfit a small dataset
            crop_dir = dir + 'input_small/' + str(small_num) + '/' # try to overfit a small dataset

        gt_nms = os.listdir(gt_dir)
        for nm in gt_nms:
            if nm.endswith('pickle'):
                gt_path = gt_dir + nm
                crop_path = crop_dir + nm.replace('pickle', 'exr')
                if os.path.exists(crop_path):
                    self.pairs.append([crop_path, gt_path])
        self.data_len = len(self.pairs)
        self.to_tensor = transforms.ToTensor()

        self.tone = util.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
        self.handle = util.PanoramaHandler()

    def __getitem__(self, index):
        training_pair = {
            "crop": None,
            "distribution": None,
            'intensity': None,
            'rgb_ratio': None,
            'ambient': None,
            # 'depth': None,
            'name': None}

        pair = self.pairs[index]
        crop_path = pair[0]

        exr = self.handle.read_hdr(crop_path)
        input, alpha = self.tone(exr)
        training_pair['crop'] = self.to_tensor(input)

        gt_path = pair[1]
        handle = open(gt_path, 'rb')
        gt = pickle.load(handle)

        training_pair['distribution'] = torch.from_numpy(gt['distribution']).float()
        training_pair['intensity'] = torch.from_numpy(np.array(gt['intensity'])).float() * alpha / 500
        training_pair['rgb_ratio'] = torch.from_numpy(gt['rgb_ratio']).float()
        training_pair['ambient'] = torch.from_numpy(gt['ambient']).float() * alpha / (128 * 256)
        # training_pair['depth'] = torch.from_numpy(gt['depth']).float()

        training_pair['name'] = gt_path.split('/')[-1].split('.pickle')[0]

        return training_pair

    def __len__(self):
        return self.data_len
