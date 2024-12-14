# 用于将整个数据集中的训练数据、测试数据和验证数据分离开
import argparse
import os
import shutil
from copy import copy

paser = argparse.ArgumentParser()
# 不同文件的原地址和目标地址
paser.add_argument("--hqsourcePath", type=str, default="E:\dataset\inpainting\CelebAMask-HQ\CelebA-HQ-img")
paser.add_argument("--hqdestPath", type=str, default="E:\dataset\inpainting\CelebAMask-HQ")

paser.add_argument("--dtdsourcePath", type=str, default="E:\dataset\inpainting\dtd\images")
paser.add_argument("--dtddestPath", type=str, default="E:\dataset\inpainting\dtd")
paser.add_argument("--dtdtxtPath", type=str, default="E:\dataset\inpainting\dtd\labels")

paser.add_argument("--sourcePath", type=str, default="E:\dataset\inpainting\CelebA\Img\img_align_celeba")
paser.add_argument("--destPath", type=str, default="E:\dataset\inpainting\CelebA")
paser.add_argument("--txtPath", type=str, default="E:\dataset\inpainting\CelebA\Eval\list_eval_partition.txt")
opt = paser.parse_args()

def _celeba():
    # 读取tex文件中的数据并判断是用于测试、训练还是验证
    train_path = os.path.join(opt.destPath,"train")
    vali_path = os.path.join(opt.destPath, "vali")
    test_path = os.path.join(opt.destPath, "test")

    test_num = 0
    vali_num = 0

    with open(opt.txtPath, 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')  # 去除文本中的换行符
            image_name, label = ann.split()
            # txt文件中不同用处的图片的名称后面的标记是不同的 0 1 2
            # 根据不同标记判定
            if label == "0":
                source_image = os.path.join(opt.sourcePath,image_name)
                dest_image = os.path.join(train_path, image_name)
                shutil.copy2(source_image, dest_image)
            elif label == "1":
                source_image = os.path.join(opt.sourcePath, image_name)
                dest_image = os.path.join(vali_path, image_name)
                shutil.copy2(source_image, dest_image)
                vali_num += 1
            else :
                source_image = os.path.join(opt.sourcePath, image_name)
                dest_image = os.path.join(test_path, image_name)
                shutil.copy2(source_image, dest_image)
                test_num +=1

    print("vali_num="+str(vali_num))
    print("test_num="+str(test_num))

def _celebaHQ():
    # HQ中图像训练、验证、测试的比例为8:1:1
    train_path = os.path.join(opt.hqdestPath, "train")
    vali_path = os.path.join(opt.hqdestPath, "vali")
    test_path = os.path.join(opt.hqdestPath, "test")

    for dir, dirs , imagename in os.walk(opt.hqsourcePath):
        imagename = sorted(imagename)
        num = len(imagename)
        for i in range(num):
            name = imagename[i]
            source_image = os.path.join(opt.hqsourcePath, name)
            label , _ = name.split(".")

            if int(label) < num/10*8:
                dest_image = os.path.join(train_path, name)
            elif int(label) < num/10*9:
                dest_image = os.path.join(vali_path, name)
            else:
                dest_image = os.path.join(test_path, name)

            shutil.copy2(source_image, dest_image)


def _dtd():
    # 读取所有txt文件中的图像地址 制作为三个列表
    train_path = os.path.join(opt.dtddestPath,"train")
    vali_path = os.path.join(opt.dtddestPath, "vali")
    test_path = os.path.join(opt.dtddestPath, "test")

    label_list = ["train1.txt","val1.txt","test1.txt", ]

    for i in range(len(label_list)):
        txtname = label_list[i]
        txt_path = os.path.join(opt.dtdtxtPath, txtname)
        with open(txt_path, 'r', encoding='utf-8') as f:
            for ann in f.readlines():
                ann = ann.strip('\n')  # 去除文本中的换行符
                if "train" in txtname:
                    source_image = os.path.join(opt.dtdsourcePath, ann)
                    _, image_name = ann.split("/")
                    dest_image = os.path.join(train_path, image_name)
                elif ("test" in txtname):
                    source_image = os.path.join(opt.dtdsourcePath, ann)
                    _, image_name = ann.split("/")
                    dest_image = os.path.join(test_path, image_name)
                else:
                    source_image = os.path.join(opt.dtdsourcePath, ann)
                    _, image_name = ann.split("/")
                    dest_image = os.path.join(vali_path, image_name)
                shutil.copy2(source_image, dest_image)


if __name__ == "__main__":
    _celeba()
    # _celebaHQ()
    # _dtd()