import lpips
import torch
import os

import numpy as np
import argparse
import matplotlib.pyplot as plt

from glob import glob
from ntpath import basename
# from scipy.misc import imread
# from skimage.measure import compare_ssim
# from skimage.measure import compare_psnr
from PIL import Image
from imageio import imread
# from skimage.color import rgb2gray


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    # parser.add_argument('--data-path',default='E:\dataset\inpainting\Celeba\celeba_test', help='Path to ground truth data', type=str)
    parser.add_argument('--data-path', default=r'F:\datasets\inpainting\paris\vali',help='Path to ground truth data', type=str)
    # parser.add_argument('--data-path', default='E:\dataset\inpainting\place2\place_test',help='Path to ground truth data', type=str)
    # parser.add_argument('--output-path', default='./checkpoints/places2/result6', help='Path to output data', type=str)
    # parser.add_argument('--output-path',default='./checkpoints/celeba/result2' , help='Path to output data', type=str)
    parser.add_argument('--output-path', default=r'E:\5.机器学习project\图像修复\经典\RFR-Inpainting-master\checkpoints\psv\result1', help='Path to output data', type=str)
    # parser.add_argument('--output-path', default='./checkpoints/place2/result6', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args

args = parse_args()
for arg in vars(args):
    # 打印参数值
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

lpips = []
names = []
index = 1

def resize( img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # 先整成正方形
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    # img = scipy.misc.imresize(img, [height, width])  旧版本，过时
    img = Image.fromarray(img)  # 采用替代方法
    size = tuple((np.array([height, width]).astype(int)))
    img = np.array(img.resize(size))

    return img

for root, dirs ,files,in os.walk(path_true):
    for dir in dirs:
        files = list(glob(path_true + '/'+dir + '/*.jpg')) + list(glob(path_true + '/'+dir +  '/*.png'))
# for root, files, in os.walk(path_true):

    for fn in sorted(files):
        name = basename(str(fn))   # 得到地址中的文件名
        names.append(name)

    # 将图片转化为0-1范围

        img_pred = (imread(path_pred + '/' + basename(str(fn))) / 255.0).astype(np.float32)
        w,h = img_pred.shape[0:2]
        # img_gt =imread(path_true + '/'+dir+ '/' + basename(str(fn)))
        img_gt = imread(path_true + '/' + basename(str(fn)))
        img_gt = resize(img_gt,w,h)
        img_gt = (img_gt/255.0).astype(np.float32)

        # img_gt = rgb2gray(img_gt)  # 去掉维度通道，将三通道变为一通道
        # img_pred = rgb2gray(img_pred)
        img_gt = torch.tensor(np.expand_dims(img_gt.transpose(2,0,1),axis=0)).to(device)
        img_pred = torch.tensor(np.expand_dims(img_pred.transpose(2,0,1),axis=0)).to(device)



        lpips.append(loss_fn_alex(img_gt, img_pred).detach().cpu().numpy().squeeze())
    # lpips = [i for i.cpu().n in lpips]

    if np.mod(index, 100) == 0:
        print(
            str(index) + ' images processed',
            "lpips: %.4f" % round(np.mean(lpips), 4),  # round，按指定精度四舍五入
        )
    index += 1
# 以字典方式存储到文件中  data = np.load() data['psnr']方式取出
np.savez(args.output_path + '/metrics.npz', lpia=lpips, names=names)
print(
    "lpips: %.4f" % round(np.mean(lpips), 4),
    "lpips Variance: %.4f" % round(np.var(lpips), 4),

)


img0 = torch.zeros(1,3,64,64) # image should be RGB, IMPORTANT: normalized to [-1,1]
img1 = torch.zeros(1,3,64,64)
d = loss_fn_alex(img0, img1)