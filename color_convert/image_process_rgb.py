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
        # 保存每个通道的图片
        cv2.imwrite('bgr_blue_channel.jpg', blue_channel)
        cv2.imwrite('bgr_green_channel.jpg', green_channel)
        cv2.imwrite('bgr_red_channel.jpg', red_channel)

        print("通道分离完毕，每个通道的图像已保存。")

        # 阈值分割
        for i in [10, 20, 30, 40, 50, 60]:
            Lower = np.array([0, 0, 100])  # BGR
            Upper = np.array([i, i, 255])  # BGR
            Binary = cv2.inRange(image, Lower, Upper)
            cv2.imwrite('bgr_channel_threshold_split_{0}.jpg'.format(i), Binary)

        # rgb->yuv
        # yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # cv2.imwrite("yuv_strawberry.jpg", yuv_image)
        # Y,U,V = cv2.split(yuv_image)
        # Lowerred0 = np.array([0, 100, 80])
        # Upperred0 = np.array([255, 120, 125])
        # yuv_leaf_mask = cv2.inRange(yuv_image, Lowerred0, Upperred0)
        # cv2.imwrite("yuv_leaf_mask.jpg", yuv_leaf_mask)
        
        # Strawberry = cv2.inRange(V,170,255)
        # cv2.imwrite("Strawberry.jpg", Strawberry)
        # cv2.waitKey(0)
if __name__ == "__main__":
    main()
