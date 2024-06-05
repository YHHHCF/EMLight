import os
import os.path
import numpy as np
import shutil

n_val = 256
n_test = 256
n_small_list = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

input_dir = 'crop/' # all input images

test_input_dir = 'input_test/' # 256 images randomly sampled from all input images for testing
train_input_dir = 'input_train/' # the rest of the images are for training
val_input_dir = 'input_val/' # 256 images randomly sampled from all input images for testing
small_input_dir = 'input_small/' # a small amount of images randomly sampled from training images for small dataset overfit

input_nms = os.listdir(input_dir)
n_train = len(input_nms) - n_test - n_val

idx = np.arange(len(input_nms))
np.random.shuffle(idx)

val_idx = idx[:n_val]
test_idx = idx[n_val:n_val+n_test]
train_idx = idx[n_val+n_test:]

print("Total, train, val, test:", len(input_nms), len(train_idx), len(val_idx), len(test_idx))

for idx in train_idx:
	file_name = input_nms[idx]
	shutil.copyfile(input_dir + file_name, train_input_dir + file_name)

for idx in val_idx:
	file_name = input_nms[idx]
	shutil.copyfile(input_dir + file_name, val_input_dir + file_name)

for idx in test_idx:
	file_name = input_nms[idx]
	shutil.copyfile(input_dir + file_name, test_input_dir + file_name)

for n_small in n_small_list:
	dir = small_input_dir + str(n_small) + '/'
	small_idx = train_idx[:n_small]
	for idx in small_idx:
		file_name = input_nms[idx]
		shutil.copyfile(input_dir + file_name, dir + file_name)
