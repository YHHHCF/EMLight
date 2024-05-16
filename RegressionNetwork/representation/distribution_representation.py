import numpy as np
from PIL import Image
import pickle
import os.path
import imageio
import vtk
from vtk.util import numpy_support
import torch
import util

imageio.plugins.freeimage.download()

class extract_mesh():
    def __init__(self, h=128, w=256, ln=64):
        self.h, self.w = h, w
        steradian = np.linspace(0, h, num=h, endpoint=False) + 0.5
        steradian = np.sin(steradian / h * np.pi)
        steradian = np.tile(steradian.transpose(), (w, 1))
        steradian = steradian.transpose()
        self.steradian = steradian[..., np.newaxis]

        y_ = np.linspace(0, np.pi, num=h)  # + np.pi / h
        x_ = np.linspace(0, 2 * np.pi, num=w)  # + np.pi * 2 / w
        X, Y = np.meshgrid(x_, y_)
        Y = Y.reshape((-1, 1))
        X = X.reshape((-1, 1))
        phi_theta = np.stack((X, Y), axis=1)
        xyz = util.polar_to_cartesian(phi_theta)
        xyz = xyz.reshape((h, w, 3))  # 128, 256, 3
        xyz = np.expand_dims(xyz, axis=2)
        self.xyz = np.repeat(xyz, ln, axis=2)
        self.anchors = util.sphere_points(ln)

        dis_mat = np.linalg.norm((self.xyz - self.anchors), axis=-1)
        self.idx = np.argsort(dis_mat, axis=-1)[:, :, 0]
        self.ln, _ = self.anchors.shape

    def compute(self, hdr):
        hdr = self.steradian * hdr
        hdr_intensity = 0.3 * hdr[..., 0] + 0.59 * hdr[..., 1] + 0.11 * hdr[..., 2]
        max_intensity_ind = np.unravel_index(np.argmax(hdr_intensity, axis=None), hdr_intensity.shape)
        max_intensity = hdr_intensity[max_intensity_ind]
        map = hdr_intensity > (max_intensity * 0.05)
        map = np.expand_dims(map, axis=-1)
        light = hdr * map
        remain = hdr * (1 - map)

        ambient = remain.sum(axis=(0, 1))    #mean(axis=0).mean(axis=0)
        anchors = np.zeros((self.ln, 3))

        for i in range(self.ln):
            mask = self.idx == i
            mask = np.expand_dims(mask, -1)
            anchors[i] = (light * mask).sum(axis=(0, 1))

        anchors_engergy = 0.3 * anchors[..., 0] + 0.59 * anchors[..., 1] + 0.11 * anchors[..., 2]
        distribution = anchors_engergy / anchors_engergy.sum()
        anchors_rgb = anchors.sum(0)   # energy
        intensity = np.linalg.norm(anchors_rgb)
        rgb_ratio = anchors_rgb / intensity
        # distribution = anchors / intensity

        parametric_lights = {"distribution": distribution,
                             'intensity': intensity,
                             'rgb_ratio': rgb_ratio,
                             'ambient': ambient}
        return parametric_lights, map

bs_dir = '/Users/erbao/Life/CS231N/project/EMLight/Dataset/LavalIndoor/'
hdr_dir = bs_dir + 'Stage1/warpedHDROutputs/'
sv_dir = bs_dir + 'Stage2/params/'
nms = os.listdir(hdr_dir)
nms = nms[:1]
ln = 128

extractor = extract_mesh(ln=ln)

i = 0
for nm in nms:
    if nm.endswith('.exr'):
        hdr_path = hdr_dir + nm

        h = util.PanoramaHandler()
        hdr = h.read_exr(hdr_path) # (128, 256, 3)

        # param["distribution"], (128,), sum to 1
        # param["intensity"], scaler value
        # param["rgb_ratio"], (3,)
        # param["ambient"], (3,)
        # map, (128, 256, 1), true if the entry is part of light source
        param, map = extractor.compute(hdr)

        with open((sv_dir + os.path.basename(hdr_path).replace('exr', 'pickle')), 'wb') as handle:
            pickle.dump(param, handle, protocol=pickle.HIGHEST_PROTOCOL)
        i += 1
        print (i)

        # dirs = util.sphere_points(ln)
        # dirs = torch.from_numpy(dirs)
        # dirs = dirs.view(1, ln*3).cuda().float()
        #
        # size = torch.ones((1, ln)).cuda().float() * 0.0025
        # intensity = torch.from_numpy(np.array(para['intensity'])).float().cuda()
        # intensity = intensity.view(1, 1, 1).repeat(1, ln, 3).cuda()
        #
        # rgb_ratio = torch.from_numpy(np.array(para['rgb_ratio'])).float().cuda()
        # rgb_ratio = rgb_ratio.view(1, 1, 3).repeat(1, ln, 1).cuda()
        #
        # distribution = torch.from_numpy(para['distribution']).cuda().float()
        # distribution = distribution.view(1, ln, 1).repeat(1, 1, 3)
        #
        # light_rec = distribution * intensity * rgb_ratio
        # light_rec = light_rec.contiguous().view(1, ln*3)
        #
        # env = util.convert_to_panorama(dirs, size, light_rec)
        # env = env.detach().cpu().numpy()[0]
        # env = util.tonemapping(env) * 255.0
        # im = np.transpose(env, (1, 2, 0))
        # im = Image.fromarray(im.astype('uint8'))
        #
        # nm_ = nm.split('.')[0]
        # im.save('./tmp/{}_rec.png'.format(nm_))
        #
        # gt = util.tonemapping(hdr) * 255.0
        # gt = Image.fromarray(gt.astype('uint8'))
        # gt.save('./tmp/{}_gt.png'.format(nm_))
        #
        # light = util.tonemapping(hdr) * 255.0 * map
        # light = Image.fromarray(light.astype('uint8'))
        # light.save('./tmp/{}_light.png'.format(nm_))
        # print (1/0)
