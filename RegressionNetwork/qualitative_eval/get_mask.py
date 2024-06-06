import cv2
from matplotlib import pyplot as plt

img = cv2.imread('rendered/env_pred_Original_DenseNet_diffuse.png')

img[img < 20] = 0 # Blender renders some small numbers in the black areas
img[img > 0] = 1
# plt.imshow(img)
# plt.show()

cv2.imwrite('mask.png', img)
