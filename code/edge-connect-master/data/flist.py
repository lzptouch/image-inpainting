import os
import argparse
import numpy as np

"""
将各个数据集中的图片地址导入到一个flist文件中，便于访问
"""

parser = argparse.ArgumentParser()
parser.add_argument('--path', default=r'E:\dataset\inpainting',type=str, help='path to the dataset')
args = parser.parse_args()

datasets = [
            # "places2",
            # "celeba",
            #"dtd",
            #"celebahq",
            "paris_"
            ]
type_ = [
        "train",
         "vali",
         # "test",
        # "test_stru",
        #  "vali_stru",
        #  "train_stru"
         ]

# datasets = ["mask"]
# type_ = ["train", "mask1","mask2", "mask3", "mask4", "mask5", "mask6"]

def flist(imge_dir,flist_file):
    ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
    images = []
    for root, dirs, files in os.walk(imge_dir):
        print('loading ' + root)
        for file in files:
            if os.path.splitext(file)[1].upper() in ext:
                file_dir = os.path.join(root,file)
                images.append(file_dir)

    images = sorted(images)
    np.savetxt(flist_file, images, fmt='%s')

if __name__=="__main__":
    path = args.path
    for i in datasets:
        for j in type_:
            args.path = os.path.join(path, i, j)
            args.output = os.path.join(path, i, j) + ".flist"
            flist(args.path,args.output)