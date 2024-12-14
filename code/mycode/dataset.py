import os
import glob
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from PIL import Image
#from scipy.misc import imread
from imageio import imread
from utils import create_mask


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, mask_flist, augment=False, training=True):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.mask_data = self.load_flist(mask_flist)
        self.input_size = config.INPUT_SIZE


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
        img = imread(self.data[index])
        if size != 0:
            img = self.resize(img, size, size)
        mask = self.load_mask(img, index)

        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]
        return (self.to_tensor(img),self.to_tensor(mask))

    def load_mask(self, img, index):

        imgh, imgw = img.shape[0:2]
        mask_type = self.mask

        # external + random block
        if mask_type == 4:
            mask_type = 1 if np.random.binomial(1, 0.5) == 1 else 3

        # external + random block + half
        elif mask_type == 5:
            mask_type = np.random.randint(1, 4)

        # random block
        if mask_type == 1:
            return create_mask(imgw, imgh, imgw // 2, imgh // 2)

        # half
        if mask_type == 2:
            # randomly choose right or left
            return create_mask(imgw, imgh, imgw // 2, imgh, 0 if random.random() < 0.5 else imgw // 2, 0)

        # external
        if mask_type == 3:
            if index==len(self.mask_data):
                index = index%len(self.mask_data)

            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            # print(mask_index)
            mask = self.resize(mask, imgh, imgw)
            mask = (mask > 0).astype(np.uint8) * 255       # threshold due to interpolation
            return mask

        # test mode: load mask non random
        if mask_type == 6:
            if index >= len(self.mask_data):
                index = (index % len(self.mask_data))
            mask = imread(self.mask_data[index])
            mask = self.resize(mask, imgh, imgw, centerCrop=False)
            # mask = rgb2gray(mask)
            mask = (mask > 0).astype(np.uint8) * 255
            return mask

    def to_tensor(self, img):
        # Image->np.Arrary->tensor
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def resize(self, img, height, width, centerCrop=True):
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

    def load_flist(self, flist):
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

    def create_iterator(self, batch_size):
            while True:
                sample_loader = DataLoader(
                    dataset=self,
                    batch_size=batch_size,
                    drop_last=True
                )

                for item in sample_loader:
                    yield item


    def load(self, image_path, mask_path):
        img = imread(image_path)

        mask = imread(mask_path)
        mask = self.resize(mask, 256, 256, centerCrop=False)

        mask = (mask > 0).astype(np.uint8) * 255
        # augment data
        if self.augment and np.random.binomial(1, 0.5) > 0:
            img = img[:, ::-1, ...]
            mask = mask[:, ::-1, ...]
        return (self.to_tensor(img), self.to_tensor(mask))

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}

