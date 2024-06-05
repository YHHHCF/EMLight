import cv2
import os.path
import matplotlib.pyplot as plt
import numpy as np

experiments = ['paper_pretrained', 'Original_DenseNet', 'Original_DenseNet_Relu_Norm', 'semantic_model']

for experiment in experiments:
	eval_dir = './results/' + experiment + '/results/rendered/'
	nms = os.listdir(eval_dir)

	categories = ['diffuse', 'matte', 'mirror']
	count = np.zeros(3)
	RMSE = np.zeros(3)
	mask_num = 6562 # number of pixels of the sphere mask

	for nm in nms:
		for c in range(3):
			category = categories[c]
			suffix = 'gt_' + category
			if suffix in nm:
				gt_path = eval_dir + nm
				gt_rendered = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB)
				pred_path = eval_dir + nm.replace('gt', 'pred')
				pred_rendered = cv2.cvtColor(cv2.imread(pred_path), cv2.COLOR_BGR2RGB)
				# plt.imshow(gt_rendered)
				# plt.show()
				# plt.imshow(pred_rendered)
				# plt.show()
				RMSE[c] += np.sqrt(np.sum((pred_rendered - gt_rendered) ** 2) / mask_num) / 255
				count[c] += 1
				break
	RMSE = RMSE / count
	print("========================")
	print("Experiment:", experiment)
	for i in range(3):
		print(categories[i], RMSE[i])
