import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import data
from torch.optim import lr_scheduler
import numpy as np
import pickle
from util import PanoramaHandler, TonemapHDR, tonemapping
from PIL import Image
import util
import DenseNet
from torchvision import transforms
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ln = 96
tone = util.TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.9)
transform = transforms.ToTensor()

# Model = DenseNet.DenseNet().to(device)
# Model = DenseNet.OriginalDenseNet().to(device)
Model = DenseNet.SemanticsDenseNet().to(device)

Model.load_state_dict(torch.load("./checkpoints/semantic_model/latest_net.pth", map_location=device))

img = cv2.cvtColor(cv2.imread('qualitative_eval/resized.jpg'), cv2.COLOR_BGR2RGB).astype(np.uint8)
input = transform(img)
input = input.view(1, 3, 192, 256).to(device)

sem = cv2.cvtColor(cv2.imread('qualitative_eval/semantics.jpg'), cv2.COLOR_BGR2RGB).astype(np.uint8)
semantics = transform(sem)
semantics = semantics.view(1, 3, 192, 256).to(device)

pred = Model(input, semantics)
# pred = Model(input)

dist_pred = pred['distribution']
rgb_ratio_pred = pred['rgb_ratio']
ambient_pred = pred['ambient']

dist_pred[dist_pred < 0] = 0
dist_pred[dist_pred > 1] = 1

rgb_ratio_pred[rgb_ratio_pred < 0] = 0
rgb_ratio_pred[rgb_ratio_pred > 1] = 1

ambient_pred[ambient_pred < 0] = 0

dist_pred = dist_pred.view(-1, ln, 1) # (N=16, 96, 1)

dirs = util.sphere_points(ln)
dirs = torch.from_numpy(dirs).float()
dirs = dirs.view(1, ln * 3).to(device)

size = torch.ones((1, ln)).to(device) * 0.0025

intensity_gt = torch.ones(1, ln, 1).to(device) * 500
dist_pred = dist_pred[0].view(1, ln, 1).repeat(1, 1, 3)
rgb_ratio_pred = rgb_ratio_pred[0].view(1, 1, 3).repeat(1, ln, 1)

# use intensity from ground truth here, as mentioned in the writeup
light_pred = (dist_pred * intensity_gt * rgb_ratio_pred).view(1, ln * 3)
env_pred = util.convert_to_panorama(dirs, size, light_pred)
env_pred = np.squeeze(env_pred[0].detach().cpu().numpy())
env_pred = tone(env_pred)[0].transpose((1, 2, 0)).astype('float32') * 255.0

env_pred_img = Image.fromarray(env_pred.astype('uint8'))

# crop = np.squeeze(input.detach().cpu().numpy()).transpose((1, 2, 0)) * 255.0
# crop = Image.fromarray(crop.astype('uint8')).resize((256, 256))

# im = np.hstack((np.array(crop), np.array(env_gt_pred)))
# im = Image.fromarray(im.astype('uint8'))
env_pred_img.save('qualitative_eval/env_pred_semantic_model.jpg')
util.write_exr('qualitative_eval/env_pred_semantic_model.exr', env_pred)
