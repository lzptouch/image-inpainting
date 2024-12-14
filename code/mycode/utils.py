import glob
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


def stitch_images(inputs, *outputs, img_per_row=2):
    gap = 5
    columns = len(outputs) + 1

    width, height = inputs[0][:, :, 0].shape

    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(len(inputs) / img_per_row)))
    images = [inputs, *outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img
def stitch_images_(outputs, img_per_row=2):
    gap = 5  # 图片间的间隔
    columns = len(outputs[0])

    width, height = outputs[0][0].shape

    img = Image.new('RGB', (width * img_per_row * columns + gap * (img_per_row - 1), height * int(columns)+ gap * (columns - 1)))
    images = [*outputs]

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)

def load_flist(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]

    return []

def resize(img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0:2]

    if centerCrop and imgh != imgw:
        # 先整成正方形
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]

    img = Image.fromarray(img)   # 采用替代方法
    size = tuple((np.array([height, width]) .astype(int)))
    img = np.array(img.resize(size))


    return img

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

# -*- coding: UTF-8 -*-
from struct import unpack


# 读取并存储 bmp 文件
class ReadBMPFile:
    def __init__(self, file):
        # file = open(filePath, "rb")
        # 读取 bmp 文件的文件头    14 字节
        self.bfType = unpack("<h", file.read(2))[0]  # 0x4d42 对应BM 表示这是Windows支持的位图格式
        self.bfSize = unpack("<i", file.read(4))[0]  # 位图文件大小
        self.bfReserved1 = unpack("<h", file.read(2))[0]  # 保留字段 必须设为 0
        self.bfReserved2 = unpack("<h", file.read(2))[0]  # 保留字段 必须设为 0
        self.bfOffBits = unpack("<i", file.read(4))[0]  # 偏移量 从文件头到位图数据需偏移多少字节（位图信息头、调色板长度等不是固定的，这时就需要这个参数了）
        # 读取 bmp 文件的位图信息头 40 字节
        self.biSize = unpack("<i", file.read(4))[0]  # 所需要的字节数
        self.biWidth = unpack("<i", file.read(4))[0]  # 图像的宽度 单位 像素
        self.biHeight = unpack("<i", file.read(4))[0]  # 图像的高度 单位 像素
        self.biPlanes = unpack("<h", file.read(2))[0]  # 说明颜色平面数 总设为 1
        self.biBitCount = unpack("<h", file.read(2))[0]  # 说明比特数

        self.biCompression = unpack("<i", file.read(4))[0]  # 图像压缩的数据类型
        self.biSizeImage = unpack("<i", file.read(4))[0]  # 图像大小
        self.biXPelsPerMeter = unpack("<i", file.read(4))[0]  # 水平分辨率
        self.biYPelsPerMeter = unpack("<i", file.read(4))[0]  # 垂直分辨率
        self.biClrUsed = unpack("<i", file.read(4))[0]  # 实际使用的彩色表中的颜色索引数
        self.biClrImportant = unpack("<i", file.read(4))[0]  # 对图像显示有重要影响的颜色索引的数目
        self.bmp_data = []

        if self.biBitCount != 24:
            print("输入的图片比特值为 ：" + str(self.biBitCount) + "\t 与程序不匹配")

        for height in range(self.biHeight):
            bmp_data_row = []
            # 四字节填充位检测
            count = 0
            for width in range(self.biWidth):
                bmp_data_row.append(
                    [unpack("<B", file.read(1))[0], unpack("<B", file.read(1))[0], unpack("<B", file.read(1))[0]])
                count = count + 3
            # bmp 四字节对齐原则
            while count % 4 != 0:
                file.read(1)
                count = count + 1
            self.bmp_data.append(bmp_data_row)
        self.bmp_data.reverse()
        file.close()
        # R, G, B 三个通道
        self.R = []
        self.G = []
        self.B = []

        for row in range(self.biHeight):
            R_row = []
            G_row = []
            B_row = []
            for col in range(self.biWidth):
                B_row.append(self.bmp_data[row][col][0])
                G_row.append(self.bmp_data[row][col][1])
                R_row.append(self.bmp_data[row][col][2])
            self.B.append(B_row)
            self.G.append(G_row)
            self.R.append(R_row)
