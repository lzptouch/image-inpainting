import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default=r'./mask',type=str, help='path to the dataset')
parser.add_argument('--output', default=r'./mask.flist',type=str, help='path to the file list')
args = parser.parse_args()

ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
images = []
for root, dirs, files in os.walk(args.path):
    print('loading ' + root)
    for file in files:
        if os.path.splitext(file)[1].upper() in ext:
            #print(root)
            #print(file)

            #images.append(file_dir)
            images.append(os.path.join(root, file))

images = sorted(images)
np.savetxt(args.output, images, fmt='%s')
# 文件名能够在一个目录中，做到有序，使得在数据集的数据经过多种处理后能够做到成对访问