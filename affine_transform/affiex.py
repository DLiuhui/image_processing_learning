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
    # crop matrix, mapping the center of bbox to the center of croped image
    crop_mat = np.array(
        [[1.0, 1.0, -crop_param[0]],
         [1.0, 1.0, -crop_param[1]]], np.float32)

    crop_out = img[crop_param[1]:crop_param[1] + crop_param[3], crop_param[0]:crop_param[0] + crop_param[2]]
    cv2.imwrite(os.path.join(args.out_path, "cropped.png"), crop_out)

    out_img = cv2.warpAffine(img, crop_mat, (crop_param[3], crop_param[2]))
    cv2.imwrite(os.path.join(args.out_path, "affine_cropped.png"), out_img)

    # resize
    out_wh = [int(e) for e in args.output_size.split(',')]
    scale_x = float(out_wh[0]) / crop_param[2]
    scale_y = float(out_wh[1]) / crop_param[3]
    # scale matrix
    scale_mat = np.array(
        [[scale_x, 1.0, 0],
         [1.0, scale_y, 0]], np.float32)

    resize_out = cv2.resize(crop_out, (out_wh[0], out_wh[1]), cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(args.out_path, "resized.png"), crop_out)

    out_img = cv2.warpAffine(img, crop_mat, (crop_param[3], crop_param[2]))
    cv2.imwrite(os.path.join(args.out_path, "affine_cropped.png"), out_img)

    # shift matrix 
    shift_mat1 = np.zeros((3, 3), np.float32)
    shift_mat1[0][0] = 1
    shift_mat1[1][1] = 1
    shift_mat1[2][2] = 1

    shift_mat1[0][2] = -(out_wh[0] / 2)
    shift_mat1[1][2] = -(out_wh[1] / 2)

    # rotate matrix
    rotate_mat = np.zeros((3, 3), np.float32)
    rotate_mat[0][0] = 1
    rotate_mat[1][1] = 1
    rotate_mat[2][2] = 1

    angle = args.rotate_angle / 180.0 * 3.14159265358979323846
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rotate_mat[0][0] = cos_a
    rotate_mat[0][1] = sin_a
    rotate_mat[1][0] = -sin_a
    rotate_mat[1][1] = cos_a

    # shear matrix
    shear_mat = np.zeros((3, 3), np.float32)
    shear_mat[0][0] = 1
    shear_mat[1][1] = 1
    shear_mat[2][2] = 1

    shear_x = args.shear_factor

    shear_mat[0][1] = shear_x
    shear_mat[1][0] = shear_x

    # shift matrix 
    shift_mat2 = np.zeros((2, 3), np.float32)
    shift_mat2[0][0] = 1
    shift_mat2[1][1] = 1

    shift_mat2[0][2] = out_wh[0] / 2
    shift_mat2[1][2] = out_wh[1] / 2

    tran_mat = cv2.gemm(shift_mat2, shear_mat, 1, None, 0) 
    tran_mat = cv2.gemm(tran_mat, rotate_mat, 1, None, 0) 
    tran_mat = cv2.gemm(tran_mat, shift_mat1, 1, None, 0)
    tran_mat = cv2.gemm(tran_mat, scale_mat, 1, None, 0) 
    tran_mat = cv2.gemm(tran_mat, crop_mat, 1, None, 0) 

    out = cv2.warpAffine(img, tran_mat, (img_w, img_h))

    # cv2.imshow('original image', img)
    # cv2.waitKey(0)
    # cv2.imshow('cropped image', out)
    # cv2.waitKey(0)
    cv2.imwrite(os.path.join(args.out_path, "final.png"), out)
