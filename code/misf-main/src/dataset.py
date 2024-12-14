import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from .utils import create_mask, resize


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, flist, mask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.input_size = config.INPUT_SIZE
        # mask和edge 可以有不同的数据来源
        # mask: 随机掩码来自于文件 固定位置固定大小的掩码可以手动生成
        # edge：来自于训练和随机生成
        self.edge = config.EDGE
        self.mask = config.MASK
        self.nms = config.NMS
        self.sigma = config.SIGMA
        # 导入函数
        self.resize = resize
        # 执行 以下函数
        self.data = self.load_flist(flist)
        self.mask_data = self.load_flist(mask_flist)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)
        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)

    def load_item(self, index):
        size = self.input_size
        # load image
        img = imread(self.data[index])
        # gray to rgb
        if len(img.shape) < 3:
            # np 将三个相同部分重叠起来
            img = gray2rgb(img)
        # resize/crop if needed
        if size != 0:
            img = self.resize(img, size, size)

        # 乘以一个向量
        # 灰度图片用于生成edge
        img_gray = rgb2gray(img)
        # load mask
        mask = self.load_mask(img, index)
        # load edge

        edge = self.load_edge(img_gray, index, mask)
        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            img_gray = img_gray[:, ::-1, ...]
            edge = edge[:, ::-1, ...]
            mask = mask[:, ::-1, ...]
        return self.to_tensor(img), self.to_tensor(img_gray), self.to_tensor(edge), self.to_tensor(mask)

    def load_edge(self, img, index, mask):
        sigma = self.sigma
        mask = None if self.training else (1 - mask / 255).astype(np.bool)
        # canny
        if self.edge == 1:
            # no edge
            if sigma == -1:
                return np.zeros(img.shape).astype(np.float)
            # random sigma
            if sigma == 0:
                sigma = random.randint(1, 4)
            return canny(img, sigma=sigma, mask=mask).astype(np.float)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)
        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)
        # center
        if mask_type == 3:
            return create_mask(imgw, imgh, imgw // 2, imgh, imgw//4, imgh//4)
        # external
        if mask_type == 4:
            if index>=len(self.mask_data):
                index = index%len(self.mask_data)-2
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgw, imgh, centerCrop=False)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist
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

    def create_iterator(self, batch_size):  #创建数据集加载后的迭代器
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )
            for item in sample_loader:
                yield item
