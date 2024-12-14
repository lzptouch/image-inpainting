import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import argparse
import matplotlib.pyplot as plt
from glob import glob
from ntpath import basename

import torch
from imageio import imread
import imageio
from skimage.metrics import  structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.color import rgb2gray, gray2rgb
from utils import resize
from model.metrics import _EdgeAccuracy as edgeacc


def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path',default= r'E:\dataset\inpainting\places2\test',
                        help='Path to ground truth data', type=str)
    parser.add_argument('--output-path',
                        default=r'E:\5.project\图像修复\经典\misf-main\checkpoints\places2\result6',
                        help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args
args = parse_args()

for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

_path_true = r"E:\5.project\图像修复\参考\edge-connect-master\checkpoints\celeba\one\182639.jpg"
_path_pred = r"E:\5.project\图像修复\参考\edge-connect-master\checkpoints\celeba\one\image182639.jpg"

psnr = []
ssim = []
mae = []
names = []

#  整个文件夹中文件的评估
def folder():
    index = 1

    files = list(glob(path_pred + '/*.jpg')) + list(glob(path_pred + '/*.png'))
    for fn in sorted(files):
        name = basename(str(fn))
        names.append(name)
        img_pred = (imageio.v2.imread(str(fn)) / 255.0).astype(np.float32)
        img_gt = (imageio.v2.imread(path_true + '/' + basename(str(fn))) / 255.0).astype(np.float32)
        if len(img_gt.shape) < 3:
            img_gt = gray2rgb(img_gt)
        img_gt = rgb2gray(img_gt)
        img_gt = resize(img_gt, 256, 256)
        img_pred = rgb2gray(img_pred)
        # precision, recall = edgeacc(torch.tensor(img_gt ), torch.tensor(img_pred))

        psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
        ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51))
        mae.append(compare_mae(img_gt, img_pred))
        if np.mod(index, 3000) == 0:
            print(
                str(index) + ' images processed',
                "PSNR: %.4f" % round(np.mean(psnr), 4),
                "SSIM: %.4f" % round(np.mean(ssim), 4),
                "MAE: %.4f" % round(np.mean(mae), 4),
            )
        index += 1

    print(
        "PSNR: %.4f" % round(np.mean(psnr), 4),
        "PSNR Variance: %.4f" % round(np.var(psnr), 4),
        "SSIM: %.4f" % round(np.mean(ssim), 4),
        "SSIM Variance: %.4f" % round(np.var(ssim), 4),
        "MAE: %.4f" % round(np.mean(mae), 4),
        "MAE Variance: %.4f" % round(np.var(mae), 4)
    )

# 单张图片的评估
def file(_path_true, _path_pred):
    img_gt = (imread(_path_true) / 255.0).astype(np.float32)
    img_pred = (imread(_path_pred) / 255.0).astype(np.float32)
    if len(img_gt.shape) < 3:
        img_gt = gray2rgb(img_gt)
    img_gt = rgb2gray(img_gt)
    img_gt = resize(img_gt, 256, 256)
    img_pred = rgb2gray(img_pred)

    psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
    ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51))
    mae.append(compare_mae(img_gt, img_pred))

    print(
        "PSNR: %.4f" % round(np.mean(psnr), 4),
        "PSNR Variance: %.4f" % round(np.var(psnr), 4),
        "SSIM: %.4f" % round(np.mean(ssim), 4),
        "SSIM Variance: %.4f" % round(np.var(ssim), 4),
        "MAE: %.4f" % round(np.mean(mae), 4),
        "MAE Variance: %.4f" % round(np.var(mae), 4)
    )


if __name__ == "__main__":
    folder()
    # file(_path_true, _path_pred)