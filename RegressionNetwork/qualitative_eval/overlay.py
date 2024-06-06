import os
import cv2

base_dir = './rendered/'
save_dir = './overlay/'
nms = os.listdir(base_dir)

background = cv2.imread('resized.jpg')
mask = cv2.imread('mask.png')

for nm in nms:
	sphere = cv2.imread(base_dir + nm)
	overlay = background * (1 - mask) + sphere * mask
	cv2.imwrite(save_dir + nm, overlay)
