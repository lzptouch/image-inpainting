import math
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
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.color import rgb2gray

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    # parser.add_argument('--data-path', default='E:\dataset\inpainting\place2\place_test',help='Path to ground truth data', type=str)
    parser.add_argument('--data-path',default=r'F:\datasets\inpainting\paris\vali', help='Path to ground truth data', type=str)
    # parser.add_argument('--data-path',default='E:\dataset\inpainting\paris\paris_test', help='Path to ground truth data', type=str)

    # parser.add_argument('--output-path', default='./checkpoints/places2/place2s3a/result2', help='Path to output data', type=str)
    parser.add_argument('--output-path',default=r'E:\5.机器学习project\图像修复\经典\RFR-Inpainting-master\checkpoints\psv\result6' , help='Path to output data', type=str)
    # parser.add_argument('--output-path',default='./checkpoints/paris/result2' , help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args


# 计算图片像素点的平均差值
def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)


args = parse_args()
for arg in vars(args):
    # 打印参数值
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

psnr = []
ssim = []
mae = []
l1 = []
names = []
index = 1

# glob模块的主要方法就是glob, 该方法返回所有匹配的文件路径列表（list）；该方法需要一个参数用来指定匹配的路径字符串（字符串可以为绝对路径也可以为相对路径），
# 其返回的文件名只包括当前目录里的文件名，不包括子文件夹里的文件


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

        img_gt = rgb2gray(img_gt)  # 去掉维度通道，将三通道变为一通道
        img_pred = rgb2gray(img_pred)
        # a = np.linalg.norm(img_gt - img_pred)

        if args.debug != 0:
            plt.subplot('121')

            plt.imshow(img_gt)
            plt.title('Groud truth')
            plt.subplot('122')
            plt.imshow(img_pred)
            plt.title('Output')
            plt.show()

        psnr.append(compare_psnr(img_gt, img_pred, data_range=1))
        ssim.append(compare_ssim(img_gt, img_pred, data_range=1, win_size=51))
        mae.append(compare_mae(img_gt, img_pred))
        l1.append(np.linalg.norm(img_gt - img_pred)/255/3)
        if np.mod(index, 100) == 0:
            print(
                str(index) + ' images processed',
                "PSNR: %.4f" % round(np.mean(psnr), 4),  # round，按指定精度四舍五入
                "SSIM: %.4f" % round(np.mean(ssim), 4),
                "MAE: %.4f" % round(np.mean(mae), 4),
                "L1:%.4f" %round(np.mean(l1),4),
            )
        index += 1
# 以字典方式存储到文件中  data = np.load() data['psnr']方式取出
np.savez(args.output_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names)
print(
    "PSNR: %.4f" % round(np.mean(psnr), 4),
    "PSNR Variance: %.4f" % round(np.var(psnr), 4),
    "SSIM: %.4f" % round(np.mean(ssim), 4),
    "SSIM Variance: %.4f" % round(np.var(ssim), 4),
    "MAE: %.4f" % round(np.mean(mae), 4),
    "MAE Variance: %.4f" % round(np.var(mae), 4),
    "L1: %.4f" % round(np.mean(l1), 4),
    "L1 Variance: %.4f" % round(np.var(l1), 4)
)
print(psnr)


