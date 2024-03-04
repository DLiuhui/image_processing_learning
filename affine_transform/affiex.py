import cv2
import argparse
import os
import math
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image', type=str, default='./timg.jpg', help='')
    parser.add_argument('--out-path', type=str, default='./', help='')
    parser.add_argument('--crop-param', type=str, default='110,60,400,290', help='')
    parser.add_argument('--rotate-angle', type=float, default=30, help='')
    parser.add_argument('--output-size', type=str, default='300,200', help='')
    parser.add_argument('--shear-factor', type=float, default=0.1, help='')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    img = cv2.imread(args.image, 1)
    img_h, img_w = img.shape[:2]
    # cv2.imshow('original image', img)
    # cv2.waitKey(0)

    # crop
    # [左上角x, 左上角y, 裁切宽w, 裁切长h]
    crop_param = [int(e) for e in args.crop_param.split(',')]
    crop_out = img[crop_param[1]:crop_param[1] + crop_param[3], crop_param[0]:crop_param[0] + crop_param[2]]
    cv2.imwrite(os.path.join(args.out_path, "cropped.png"), crop_out)
    # crop本质上是先执行平移, 将左上角点变为crop的左上角点, 然后再保留crop-w/crop-h部分
    # crop matrix, mapping the center of bbox to the center of croped image
    crop_mat = np.array(
        [[1.0, 0.0, -crop_param[0]],
         [0.0, 1.0, -crop_param[1]],
         [0.0, 0.0, 1.0]], np.float32)
    out_img = cv2.warpAffine(img, crop_mat[:2,:], (crop_param[2], crop_param[3]))
    cv2.imwrite(os.path.join(args.out_path, "affine_cropped.png"), out_img)
    print("image-value-diff crop: opencv vs affine, max-diff ",
          np.max(out_img - crop_out), " mean-diff ", np.mean(out_img - crop_out))

    # resize
    out_wh = [int(e) for e in args.output_size.split(',')]
    resize_out = cv2.resize(crop_out, (out_wh[0], out_wh[1]), cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(args.out_path, "resized.png"), resize_out)
    # scale matrix
    scale_x = float(out_wh[0]) / crop_param[2]
    scale_y = float(out_wh[1]) / crop_param[3]
    scale_mat = np.array(
        [[scale_x, 0.0, 0.0],
         [0.0, scale_y, 0.0],
         [0.0, 0.0, 1.0]], np.float32)

    out_img = cv2.warpAffine(out_img, scale_mat[:2,:], (out_wh[0], out_wh[1]))
    cv2.imwrite(os.path.join(args.out_path, "affine_resized.png"), out_img)
    print("image-value-diff resize: opencv vs affine, max-diff ",
          np.max(out_img - resize_out), " mean-diff ", np.mean(out_img - resize_out))

    # rotate matrix
    angle = args.rotate_angle / 180.0 * np.pi
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rotate_mat = np.array(
        [[cos_a, sin_a, 0.0],
         [-sin_a, cos_a, 0.0],
         [0.0, 0.0, 1.0]], np.float32)
    out_img = cv2.warpAffine(out_img, rotate_mat[:2,:], (out_wh[0], out_wh[1]))
    cv2.imwrite(os.path.join(args.out_path, "affine_rotate.png"), out_img)

    # shear matrix
    shear_x = args.shear_factor
    shear_mat = np.array(
        [[1.0, shear_x, 0.0],
         [shear_x, 1.0, 0.0],
         [0.0, 0.0, 1.0]], np.float32)
    out_img = cv2.warpAffine(out_img, shear_mat[:2,:], (out_wh[0], out_wh[1]))
    cv2.imwrite(os.path.join(args.out_path, "affine_shear.png"), out_img)

    # # shift matrix 
    # shift_mat2 = np.zeros((2, 3), np.float32)
    # shift_mat2[0][0] = 1
    # shift_mat2[1][1] = 1

    # shift_mat2[0][2] = out_wh[0] / 2
    # shift_mat2[1][2] = out_wh[1] / 2

    # 不分步, 直接完整变换
    # 可以看到变换之后的关键字部分，完整变换和分步变换接近
    # 但是完整变换没有图像的截断
    # tran_mat = cv2.gemm(shift_mat2, shear_mat, 1, None, 0) 
    tran_mat = cv2.gemm(shear_mat, rotate_mat, 1, None, 0) 
    tran_mat = cv2.gemm(tran_mat, scale_mat, 1, None, 0)
    tran_mat = cv2.gemm(tran_mat, crop_mat, 1, None, 0)
    tran_img = cv2.warpAffine(img, tran_mat[:2,:], (out_wh[0], out_wh[1]))
    # cv2.imshow('original image', img)
    # cv2.waitKey(0)
    # cv2.imshow('cropped image', out)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(args.out_path, "affine_final.png"), tran_img)

    tran_img_origin = cv2.warpAffine(img, tran_mat[:2,:], (img.shape[1], img.shape[0]))
    cv2.imwrite(os.path.join(args.out_path, "affine_final_origin.png"), tran_img_origin)
