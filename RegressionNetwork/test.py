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
from geomloss import SamplesLoss

import imageio
imageio.plugins.freeimage.download()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
h = PanoramaHandler()
batch_size = 16
ln = 96

test_dataset = data.ParameterDataset("../Dataset/LavalIndoor/", mode="test")
dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

l2 = nn.MSELoss().to(device)
Sam_Loss = SamplesLoss("sinkhorn", p=2, blur=.025, batchsize=batch_size)

Model = DenseNet.DenseNet().to(device)
# Model = DenseNet.OriginalDenseNet().to(device)
# Model = DenseNet.SemanticsDenseNet().to(device)
load_weight = True
if load_weight:
    Model.load_state_dict(torch.load("./checkpoints/paper/latest_net.pth", map_location=device))
    print ('load trained model')
tone = util.TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.9)

for i, para in enumerate(dataloader):
    nm = para['name'][0]

    input = para['crop'].to(device)
    # semantics = para['semantics'].to(device)
    # pred = Model(input, semantics)
    pred = Model(input)

    dist_pred, dist_gt = pred['distribution'], para['distribution'].to(device) # (16, ln=96)
    intensity_gt = para['intensity'].to(device).view(-1, 1) # (16, 1)
    rgb_ratio_pred, rgb_ratio_gt = pred['rgb_ratio'], para['rgb_ratio'].to(device) # (16, 3)
    ambient_pred, ambient_gt = pred['ambient'], para['ambient'].to(device) # (16, 3)

    dist_pred[dist_pred < 0] = 0
    dist_pred[dist_pred > 1] = 1

    rgb_ratio_pred[rgb_ratio_pred < 0] = 0
    rgb_ratio_pred[rgb_ratio_pred > 1] = 1

    ambient_pred[ambient_pred < 0] = 0

    dist_pred = dist_pred.view(-1, ln, 1) # (N=16, 96, 1)
    dist_gt = dist_gt.view(-1, ln, 1) # (N=16, 96, 1)
    dist_emloss = (Sam_Loss(dist_pred, dist_gt).sum() / batch_size) * 1000.0
    dist_l2loss = (l2(dist_pred, dist_gt) / batch_size) * 1000.0
    rgb_loss = (l2(rgb_ratio_pred, rgb_ratio_gt) / batch_size) * 100.0
    ambient_loss = (l2(ambient_pred, ambient_gt) / batch_size) * 1.0

    print("dist_emloss:{:.3f}, dist_l2loss:{:.3f}, rgb_loss:{:.4f}, ambient_loss:{:.5f}"
          .format(dist_emloss.item(), dist_l2loss.item(), rgb_loss.item(), ambient_loss.item()))

    dirs = util.sphere_points(ln)
    dirs = torch.from_numpy(dirs).float()
    dirs = dirs.view(1, ln * 3).to(device)

    size = torch.ones((1, ln)).to(device) * 0.0025

    intensity_gt = intensity_gt[0].view(1, 1, 1).repeat(1, ln, 3) * 500
    dist_pred = dist_pred[0].view(1, ln, 1).repeat(1, 1, 3)
    rgb_ratio_pred = rgb_ratio_pred[0].view(1, 1, 3).repeat(1, ln, 1)

    # use intensity from ground truth here, as mentioned in the writeup
    light_pred = (dist_pred * intensity_gt * rgb_ratio_pred).view(1, ln * 3)
    env_pred = util.convert_to_panorama(dirs, size, light_pred)
    env_pred = np.squeeze(env_pred[0].detach().cpu().numpy())
    env_pred = tone(env_pred)[0].transpose((1, 2, 0)).astype('float32') * 255.0

    dist_gt = dist_gt[0].view(1, ln, 1).repeat(1, 1, 3)
    rgb_ratio_gt = rgb_ratio_gt[0].view(1, 1, 3).repeat(1, ln, 1)

    light_gt = (dist_gt * intensity_gt * rgb_ratio_gt).view(1, ln * 3)
    env_gt = util.convert_to_panorama(dirs, size, light_gt)
    env_gt = np.squeeze(env_gt[0].detach().cpu().numpy())
    env_gt = tone(env_gt)[0].transpose((1, 2, 0)).astype('float32') * 255.0
    env_gt_pred = np.vstack((env_gt, env_pred))
    env_gt_pred = Image.fromarray(env_gt_pred.astype('uint8')).resize((256, 256))

    crop = np.squeeze(input[0].detach().cpu().numpy()).transpose((1, 2, 0)) * 255.0
    crop = Image.fromarray(crop.astype('uint8')).resize((256, 256))

    im = np.hstack((np.array(crop), np.array(env_gt_pred)))
    im = Image.fromarray(im.astype('uint8'))
    im.save('./results/{}.jpg'.format(nm))

    env_gt /= 500
    env_pred /= 500

    env_gt_path = './results/{}_gt.exr'.format(nm)
    util.write_exr(env_gt_path, env_gt)

    # use intensity from ground truth here, as mentioned in the writeup
    adjust_ratio = np.max(env_gt) / np.max(env_pred)
    env_pred_path = './results/{}_pred.exr'.format(nm)
    util.write_exr(env_pred_path, env_pred * adjust_ratio)

    print (i)
