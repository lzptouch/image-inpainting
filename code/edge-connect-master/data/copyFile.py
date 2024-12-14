# 将places2文件夹下的所有子文件夹中的文件整理在一起，并命名
import os
from os.path import basename
from shutil import copy2

source_path = r"E:\dataset\inpainting\places2\train_"
dest_path = r"E:\dataset\inpainting\places2\train"

if not os.path.exists(dest_path):
    os.mkdir(dest_path)

def _copyfile(source_path, dest_path):
    # 文件夹名，子文件夹列表
    for root, dirs, files in os.walk(source_path):
        basedir = basename(root)
        for file in files:
            source = os.path.join(root, file)
            dest = os.path.join(dest_path,basedir + file)
            copy2(source, dest)

if __name__ == "__main__":
    _copyfile(source_path, dest_path)