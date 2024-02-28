import cv2
import numpy as np

def main():
    # 读取图片
    image = cv2.imread('strawberry.jpg')

    # 检查图像是否被正确载入
    if image is None:
        print("Error: 图像没有被正确载入。")
    else:
        # 分离图像通道
        blue_channel, green_channel, red_channel = cv2.split(image)
        # print(blue_channel)

        red_channel = red_channel.astype(np.int)
        green_channel = green_channel.astype(np.int)
        blue_channel = blue_channel.astype(np.int)

        yuv_cal_image = np.zeros(image.shape)
        # 巨硬方案
        # https://learn.microsoft.com/en-us/previous-versions/windows/embedded/ee490095(v=winembedded.60)?redirectedfrom=MSDN
        yuv_cal_image[:,:,0] = ((66 * red_channel + 129 * green_channel + 25 * blue_channel + 128) >> 8) + 16
        yuv_cal_image[:,:,1] = ((-38 * red_channel - 74 * green_channel + 112 * blue_channel + 128) >> 8) + 128
        yuv_cal_image[:,:,2] = ((112 * red_channel - 94 * green_channel - 18 * blue_channel + 128) >> 8) + 128
        cv2.imwrite("yuv_microsoft_cal_strawberry.jpg", yuv_cal_image)

        #　转换矩阵
        yuv_cal_image = np.zeros(image.shape)
        rgb_to_yuv_mat = np.array([
            [0.299, 0.587, 0.114],
            [-0.14713, -0.28886, 0.436],
            [0.615, -0.51499, -0.10001]])
        origin_shape = np.array([red_channel, green_channel, blue_channel]).shape
        yuv_cal_image = np.dot(rgb_to_yuv_mat, np.reshape(np.array([red_channel, green_channel, blue_channel]), (3, -1)))
        yuv_cal_image = yuv_cal_image.reshape(origin_shape).transpose((1, 2, 0))
        cv2.imwrite("yuv_cal_strawberry.jpg", yuv_cal_image)

        # rgb->yuv
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        cv2.imwrite("yuv_strawberry.jpg", yuv_image)
        Y,U,V = cv2.split(yuv_image)
        Lowerred0 = np.array([0, 100, 80])
        Upperred0 = np.array([255, 120, 125])
        yuv_leaf_mask = cv2.inRange(yuv_image, Lowerred0, Upperred0)
        cv2.imwrite("yuv_leaf_mask.jpg", yuv_leaf_mask)

        Strawberry = cv2.inRange(V,170,255)
        cv2.imwrite("yuv_strawberry_mask.jpg", Strawberry)
        # cv2.waitKey(0)
if __name__ == "__main__":
    main()
