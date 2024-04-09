import numpy as np
import cv2 as cv

img = cv.imread('p3re.png')
rows, cols = img.shape[:2]

# 原图中卡片的四个角点
pts1 = np.float32([[110, 60], [510, 60], [510, 350], [110, 350]])
# 变换后分别在左上、右上、左下、右下四个点
pts2 = np.float32([[0, 0], [rows / 2, 0], [rows, cols], [0, cols / 3]])

# 生成透视变换矩阵
M = cv.getPerspectiveTransform(pts1, pts2)  # M (3*3)
# 进行透视变换，参数3是目标图像大小
dst = cv.warpPerspective(img, M, (cols, rows))

cv.imwrite('perspective.png', dst)
