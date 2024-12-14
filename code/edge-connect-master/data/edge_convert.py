import os
import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--path', default=r'E:\dataset\inpainting',type=str, help='path to the dataset')
args = parser.parse_args()

datasets = ["places2", "celeba", "dtd", "celebahq"]
type_ = ["train", "vali", "test" ]

def edge_convetor(source,dest):
    ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF'}
    for root, dirs, files in os.walk(source):
        if not os.path.exists(dest):
            os.makedirs(dest)
        print('loading ' + root)
        for file in files:
            input_file = os.path.join(root,file)
            img = cv2.imread(input_file, 0)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            edge_image = cv2.Canny(img, 10, 50)
            output_file = os.path.join(dest,file)
            cv2.imwrite(output_file,edge_image)

if __name__=="__main__":
    path = args.path
    for i in datasets:
        for j in type_:
            args.path = os.path.join(path, i, j)
            args.outputpath = os.path.join(path, i, j) + "_edge"
            edge_convetor(args.path,args.outputpath)