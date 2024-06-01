import os
import os.path
import numpy as np
import shutil

input_dir = 'crop/' # all input images

test_input_dir = 'input_test/' # 200 images randomly sampled from all input images for testing
train_input_dir = 'input_train/' # the rest of the images are for training
small_input_dir = 'input_small/' # 256 images randomly sampled from training images for small dataset overfit

n_test = 200
n_small = 256

input_nms = os.listdir(input_dir)
n_train = len(input_nms) - n_test


idx = np.arange(len(input_nms))
np.random.shuffle(idx)

train_idx = idx[:n_train]
test_idx = idx[n_train:]
small_idx = idx[:n_small]

print("Total, train, train_small, test:", len(input_nms), len(train_idx), len(small_idx), len(test_idx))

for idx in train_idx:
	file_name = input_nms[idx]
	shutil.copyfile(input_dir + file_name, train_input_dir + file_name)

for idx in test_idx:
	file_name = input_nms[idx]
	shutil.copyfile(input_dir + file_name, test_input_dir + file_name)

for idx in small_idx:
	file_name = input_nms[idx]
	shutil.copyfile(input_dir + file_name, small_input_dir + file_name)
