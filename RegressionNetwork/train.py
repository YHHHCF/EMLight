import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import data
from torch.optim import lr_scheduler
import numpy as np
from util import PanoramaHandler, TonemapHDR, tonemapping
from PIL import Image
import util
import DenseNet
from geomloss import SamplesLoss

import imageio
imageio.plugins.freeimage.download()

h = PanoramaHandler()
batch_size = 8

save_dir = "./checkpoints"
train_dir = "../Dataset/LavalIndoor/"
hdr_train_dataset = data.ParameterDataset(train_dir, use_small=False)
dataloader = DataLoader(hdr_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = DenseNet.DenseNet().to(device)

torch.set_grad_enabled(True)
Model.train()

load_weight = True
if load_weight:
    Model.load_state_dict(torch.load("./checkpoints/latest_net.pth", map_location=device))
    print('load trained model')
util.print_model_parm_nums(Model)

lr_base = 0.0001
betas = (0.9, 0.999)
optimizer = torch.optim.Adam(Model.parameters(), lr=lr_base, betas=betas)
lr_decay_iters = 1000
l2 = nn.MSELoss().to(device)
Sam_Loss = SamplesLoss("sinkhorn", p=2, blur=.025, batchsize=batch_size)

tone = util.TonemapHDR(gamma=2.4, percentile=99, max_mapping=0.99)

ln = 96 # This was 128 in the paper

coord = util.sphere_points(ln)
coord = torch.from_numpy(coord).unsqueeze(0)
coord = coord.repeat(batch_size, 1, 1).to(device).float()


for epoch in range(0, 500):

    print('{} optim: {}'.format(epoch, optimizer.param_groups[0]['lr']))

    # lambda_G = lambda epoch: 0.5 ** (epoch // 30)

    for i, para in enumerate(dataloader):
        input = para['crop'].to(device) # (N=16, 3, 192, 256)
        pred = Model(input)

        dist_pred, dist_gt = pred['distribution'], para['distribution'].to(device) # (16, ln=96)
        intensity_pred, intensity_gt = pred['intensity'], para['intensity'].to(device).view(-1, 1) # (16, 1)
        rgb_ratio_pred, rgb_ratio_gt = pred['rgb_ratio'], para['rgb_ratio'].to(device) # (16, 3)
        ambient_pred, ambient_gt = pred['ambient'], para['ambient'].to(device) # (16, 3)

        dist_pred = dist_pred.view(-1, ln, 1) # (N=16, 96, 1)
        dist_gt = dist_gt.view(-1, ln, 1) # (N=16, 96, 1)
        dist_emloss = Sam_Loss(dist_pred, dist_gt).sum() * 1000.0
        dist_l2loss = l2(dist_pred, dist_gt) * 1000.0
        intensity_loss = l2(intensity_pred, intensity_gt) * 0.1
        rgb_loss = l2(rgb_ratio_pred, rgb_ratio_gt) * 100.0
        ambient_loss = l2(ambient_pred, ambient_gt) * 1.0

        loss = dist_emloss + dist_l2loss + intensity_loss + rgb_loss + ambient_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print("epoch {:0>3d} batch {:0>3d}, dist_emloss:{}, dist_l2loss:{}, intensity_loss:{}, rgb_loss:{}, ambient_loss:{}"
                  .format(epoch, i, dist_emloss.item(), dist_l2loss.item(), intensity_loss.item(), rgb_loss.item(), ambient_loss.item()))

        if i % 100 == 0:
            dirs = util.sphere_points(ln)
            dirs = torch.from_numpy(dirs).float()
            dirs = dirs.view(1, ln * 3).to(device)

            size = torch.ones((1, ln)).to(device).float() * 0.0025

            intensity_pred = intensity_pred[0].view(1, 1, 1).repeat(1, ln, 3) * 500
            dist_pred = dist_pred[0].view(1, ln, 1).repeat(1, 1, 3)
            rgb_ratio_pred = rgb_ratio_pred[0].view(1, 1, 3).repeat(1, ln, 1)

            light_pred = (dist_pred * intensity_pred * rgb_ratio_pred).view(1, ln * 3)
            env_pred = util.convert_to_panorama(dirs, size, light_pred)
            env_pred = np.squeeze(env_pred[0].detach().cpu().numpy())
            env_pred = tone(env_pred)[0].transpose((1, 2, 0)).astype('float32') * 255.0

            intensity_gt = intensity_gt[0].view(1, 1, 1).repeat(1, ln, 3) * 500
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
            im.save('./summary/{}_{}.jpg'.format(epoch, i))

        if i % 500 == 0:
            print('saving the latest model')
            save_filename = 'latest_net.pth'
            save_path = os.path.join(save_dir, save_filename)
            torch.save(Model.state_dict(), save_path)

            save_filename = 'latest_net.pth'
            save_path = os.path.join(save_dir, save_filename)
            torch.save(Model.state_dict(), save_path)


    if epoch % 10 == 0:
        print('saving the model at the end of epoch %d' % epoch)
        save_filename = '%s_net.pth' % epoch
        save_path = os.path.join(save_dir, save_filename)
        torch.save(Model.state_dict(), save_path)

        save_filename = 'latest_net.pth'
        save_path = os.path.join(save_dir, save_filename)
        torch.save(Model.state_dict(), save_path)
