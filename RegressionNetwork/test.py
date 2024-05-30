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
from gmloss import SamplesLoss

import imageio
imageio.plugins.freeimage.download()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
h = PanoramaHandler()
batch_size = 1

save_dir = "./checkpoints"
test_dir = "../Dataset/LavalIndoor/test/"
train_dir = "../Dataset/LavalIndoor/"
hdr_train_dataset = data.ParameterDataset(train_dir)
dataloader = DataLoader(hdr_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

Model = DenseNet.DenseNet().to(device)
load_weight = True
if load_weight:
    Model.load_state_dict(torch.load("./checkpoints/latest_net.pth"))
    print ('load trained model')
tone = util.TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.9)

for i, para in enumerate(dataloader):
    if i >= 10:
        break
    ln = 96

    nm = para['name'][0]

    input = para['crop'].to(device)
    pred = Model(input)

     # (1, ln=96)
    distribution_pred = pred['distribution']
    distribution_gt = para['distribution'].to(device)

     # (1, 3)
    rgb_ratio_pred = pred['rgb_ratio']
    rgb_ratio_gt = para['rgb_ratio'].to(device)

    # scalar
    intensity_pred = pred['intensity'] * 500
    intensity_gt = para['intensity'].to(device) * 500

    print(torch.any(distribution_pred < 0), torch.any(rgb_ratio_pred < 0), intensity_pred < 0)
    print(torch.any(distribution_gt < 0), torch.any(rgb_ratio_gt < 0), intensity_gt < 0)

    print (intensity_pred, intensity_gt)

    dirs = util.sphere_points(ln)
    dirs = torch.from_numpy(dirs).float()
    dirs = dirs.view(1, ln * 3).to(device)

    size = torch.ones((1, ln)).to(device) * 0.0025
    rgb_ratio_pred_repeat = rgb_ratio_pred[0].view(1, 1, 3).repeat(1, ln, 1) # (1, ln, 3)
    intensity_pred_repeat = intensity_pred[0].view(1, 1, 1).repeat(1, ln, 1) # (1, ln, 1)
    distribution_pred = distribution_pred[0].view(1, ln, 1) # (1, ln, 1)
    color_pred = rgb_ratio_pred_repeat * intensity_pred_repeat * distribution_pred # (1, ln, 3)
    color_pred = color_pred.view(1, ln*3) # (1, 3 * ln)

    hdr_pred = util.convert_to_panorama(dirs, size, color_pred)
    hdr_pred = np.squeeze(hdr_pred[0].detach().cpu().numpy())
    hdr_pred = np.transpose(hdr_pred, (1, 2, 0))

    ldr_pred = tone(hdr_pred)[0] * 255.0
    ldr_pred = Image.fromarray(ldr_pred.astype(np.uint8))
    ldr_pred.save('./results/{}_ldr_pred.jpg'.format(i))

    hdr_pred = hdr_pred.astype('float32')
    hdr_pred_path = './results/{}_pred.exr'.format(nm)
    util.write_exr(hdr_pred_path, hdr_pred)

    rgb_ratio = np.squeeze(rgb_ratio_pred[0].view(3).detach().cpu().numpy())
    intensity = np.squeeze(intensity_pred[0].detach().cpu().numpy())
    distribution = np.squeeze(distribution_pred[0].view(ln).detach().cpu().numpy())
    parametric_lights = {"distribution": distribution, "rgb_ratio": rgb_ratio, 'intensity': intensity}

    with open('./results/' + nm + '.pickle', 'wb') as handle:
        pickle.dump(parametric_lights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rgb_ratio_gt_repeat = rgb_ratio_gt[0].view(1, 1, 3).repeat(1, ln, 1)
    intensity_gt_repeat = intensity_gt[0].view(1, 1, 1).repeat(1, ln, 1)
    distribution_gt = distribution_gt[0].view(1, ln, 1)
    color_gt = rgb_ratio_gt * intensity_gt * distribution_gt
    color_gt = color_gt.view(1, ln * 3)

    hdr_gt = util.convert_to_panorama(dirs, size, color_gt)
    hdr_gt = np.squeeze(hdr_gt[0].detach().cpu().numpy())
    hdr_gt = np.transpose(hdr_gt, (1, 2, 0))

    ldr_gt = tone(hdr_gt)[0] * 255.0
    ldr_gt = Image.fromarray(ldr_gt.astype(np.uint8))
    ldr_gt.save('./results/{}_ldr_gt.jpg'.format(i))

    hdr_gt = hdr_gt.astype('float32')
    hdr_gt_path = './results/{}_gt.exr'.format(nm)
    util.write_exr(hdr_gt_path, hdr_gt)

    print (i)
