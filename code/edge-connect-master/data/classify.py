import glob
import os
import shutil
from ntpath import basename
from imageio import imread
import numpy as np
from PIL import Image

def load_flist(  flist):
    # 判断文件的类型，然后读取其中的图像名列表，
    if os.path.isfile(flist):
        try:
            return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        except:
            return [flist]
    return []

def resize( img, height, width, centerCrop=True):
    imgh, imgw = img.shape[0:2]
    # 长宽不相等时，裁剪较长的部分
    if centerCrop and imgh != imgw:
        side = np.minimum(imgh, imgw)
        j = (imgh - side) // 2
        i = (imgw - side) // 2
        img = img[j:j + side, i:i + side, ...]
    img = Image.fromarray(img)
    size = tuple((np.array([height, width]) .astype(int)))
    img = np.array(img.resize(size))
    return img

def mycopyfile(srcfile,dstdir):
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
    fname=basename(srcfile)
    shutil.copyfile(srcfile,os.path.join(dstdir,fname))

"""
将掩码数据按照掩码比例分为不同的6分
每份中的图片数基本相同
"""

if __name__ == "__main__":
    # flist 是记录了所有用于测试的mask文件地址的flist文件
    # path 是所有mask存放的目录地址的父文件夹地址
    flist = r"E:\Python\edge-connect-master\datasets\testing_mask_dataset.flist"
    path = r""
    masks = load_flist(flist)
    lenght = len(masks)
    x= np.ones((256,256))
    for i in range(0,lenght):
        mask = imread(masks[i])
        mask = resize(mask,256,256)
        sum = np.sum(mask)/(255*256*256)
        if sum<=0.1:
            mycopyfile(masks[i], os.path.join(path, "mask1"))
        elif sum>0.1 and sum<=0.2:
            mycopyfile(masks[i], os.path.join(path, "mask2"))
        elif sum>0.2 and sum<=0.3:
            mycopyfile(masks[i], os.path.join(path, "mask3"))
        elif sum >0.3 and sum <= 0.4:
            mycopyfile(masks[i], os.path.join(path, "mask4"))
        elif sum > 0.4 and sum <= 0.5:
            mycopyfile(masks[i], os.path.join(path, "mask5"))
        elif sum > 0.5 and sum <= 0.6:
            mycopyfile(masks[i], os.path.join(path, "mask6"))

