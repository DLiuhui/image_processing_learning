import cv2
import numpy as np

def bgr2hsv(bgr_image: np.ndarray, hsv_image: np.ndarray):
    # rgb -> hsv 实现
    # max(bgr)
    bgr_max_int = np.max(bgr_image, 2)
    # min(bgr)
    bgr_min_int = np.min(bgr_image, 2)
    # 取最大的idx
    bgr_arg_max = np.argmax(bgr_image, 2)
    # max(bgr) - min(bgr)
    delta_int = bgr_max_int - bgr_min_int
    # print(delta_int)

    # V = max(rgb)
    v_channel = bgr_max_int  # int 0~255

    # S
    zero_mask = bgr_max_int == 0
    v_channel_zero_to_one = bgr_max_int + zero_mask  # V矩阵里的0值赋值为1
    # max(rgb) 矩阵中为0的位置, 对应位置的 min(rgb) 一定是0, V-min(bgr)/V_0to1 就可以得到S
    s_channel = delta_int.astype(np.float) / v_channel_zero_to_one.astype(np.float)
    # print(s_channel)
    # float to int
    s_channel = (s_channel * 255.0).astype(np.int)
    # print(s_channel)

    # H
    delta_float_zero_to_one = ((delta_int == 0) + delta_int).astype(np.float)
    h_channel = np.zeros(bgr_image.shape[0:2], dtype=float)

    # 用python处理图像时，可能会涉及两幅图像像素值之间的加减运算
    # 这里需要注意的是图像像素值是ubyte类型，ubyte类型数据范围为0~255，若做运算出现负值或超出255，则会抛出异常
    # 异常信息 RuntimeWarning: overflow encountered in ubyte_scalars
    # 需要将image的像素转为int或者float类型
    for row in range(0, bgr_image.shape[0]):
        for col in range(0, bgr_image.shape[1]):
            if bgr_arg_max[row, col] == 0:  # B
                h_channel[row, col] = 240.0 + 60.0 * (float(bgr_image[row, col, 2]) - float(bgr_image[row, col, 1])) / delta_float_zero_to_one[row, col]
            elif bgr_arg_max[row, col] == 1:  # G
                h_channel[row, col] = 120.0 + 60.0 * (float(bgr_image[row, col, 0]) - float(bgr_image[row, col, 2])) / delta_float_zero_to_one[row, col]
            else:  # bgr_arg_max[row, col] == 2  R
                h_channel[row, col] = 60.0 * (float(bgr_image[row, col, 1]) - float(bgr_image[row, col, 0])) / delta_float_zero_to_one[row, col]
    zero_mask = (h_channel < 0.0).astype(np.float)
    h_channel = h_channel + zero_mask * 360.0
    # print(h_channel)
    # float to int
    h_channel = (h_channel / 2.0).astype(np.int)
    # print(h_channel)
    hsv_image[:,:,0] = h_channel
    hsv_image[:,:,1] = s_channel
    hsv_image[:,:,2] = v_channel

def main():
    # 读取图片
    image = cv2.imread('strawberry.jpg')

    # 检查图像是否被正确载入
    if image is None:
        print("Error: 图像没有被正确载入。")
    else:
        # 分离图像通道
        # blue_channel, green_channel, red_channel = cv2.split(image)

        # 实现 rgb->hsv
        hsv_image_cal = np.zeros(image.shape)
        bgr2hsv(image, hsv_image_cal)
        cv2.imwrite("hsv_cal_strawberry.jpg", hsv_image_cal)

        # rgb -> hsv
        hsv_image = np.zeros(image.shape)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cv2.imwrite("hsv_strawberry.jpg", hsv_image)

        print("h_channel diff num", np.sum(hsv_image_cal[:,:,0] != hsv_image[:,:,0]))
        print("s_channel diff num", np.sum(hsv_image_cal[:,:,1] != hsv_image[:,:,1]))
        print("v_channel diff num", np.sum(hsv_image_cal[:,:,2] != hsv_image[:,:,2]))
        # H, S, V = cv2.split(hsv_image)    #分离 HSV 三通道
        # 筛选绿色
        Lowerred0 = np.array([35,43,35])
        Upperred0 = np.array([77,255,255])
        hsv_leaf_mask = cv2.inRange(hsv_image, Lowerred0, Upperred0)
        cv2.imwrite("hsv_leaf_mask.jpg", hsv_leaf_mask)
        # 筛选红色 (跨区间, 需要两个mask)
        Lowerred0 = np.array([0,43,35])
        Upperred0 = np.array([10,255,255])
        hsv_strawberry_mask_0 = cv2.inRange(hsv_image, Lowerred0, Upperred0)
        Lowerred1 = np.array([156,43,35])
        Upperred1 = np.array([180,255,255])
        hsv_strawberry_mask_1 = cv2.inRange(hsv_image, Lowerred1, Upperred1)
        cv2.imwrite("hsv_strawberry_mask.jpg",
                    hsv_strawberry_mask_0 + hsv_strawberry_mask_1)
        # cv2.waitKey(0)

if __name__ == "__main__":
    main()
