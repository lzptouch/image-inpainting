import argparse
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='F:\paris_train_original',)
parser.add_argument('--destpath', type=str, default='E:/dataset/inpainting/paris_/train',)
args = parser.parse_args()


def _rename(path1,path2):
    if not os.path.exists(path2):
        os.mkdir(path2)
    index = 0
    for root,dir,files in os.walk(path1):
        for file in files:
            index += 1
            sourcefile = os.path.join(path1, file)
            destfile = os.path.join(path2, str(index)+".jpg")
            shutil.copy2(sourcefile, destfile)


if __name__ == "__main__":
    _rename(args.path,args.destpath)