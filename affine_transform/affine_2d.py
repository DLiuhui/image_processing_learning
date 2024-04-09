import cv2 as cv
import numpy as np

img = cv.imread('p3re.png')
rows, cols = img.shape[:2]

# 变换前的三个点
pts1 = np.float32([[110, 60], [510, 60], [510, 350]])
# 变换后的三个点
pts2 = np.float32([[0, 0], [rows, 0], [rows, cols]])

# 生成变换矩阵
M = cv.getAffineTransform(pts1, pts2)  # M (2*3)
# 第三个参数为dst的大小
dst = cv.warpAffine(img, M, (cols, rows))

cv.imwrite("affine_2d.png", dst)
