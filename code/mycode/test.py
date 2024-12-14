from multiprocessing import freeze_support

from imageio import imread

import dataset
import opt as opt
import os

from utils import imsave

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from Model import PasteModel
# from utils import create_dir, stitch_images, imsave, stitch_images_
import os
import random
import numpy as np
import torch
import cv2



def test():
    model.eval()
    # create_dir(model.results_path)
    # index = 0
    items = model.test_dataset.load("input_image.jpg", "mask.jpg")
    image, mask = model.cuda(*items)
    image = torch.unsqueeze(image,0).to(opt.DEVICE)
    mask = torch.unsqueeze(mask, 0).to(opt.DEVICE)

    outputs,  f32, f64, f128 = model(image, mask)
    outputs_merged = (outputs * mask) + (image * (1 - mask))
    output = model.postprocess(outputs_merged)[0]
    # path = os.path.join(model.results_path, name)
    imsave(output, "model.jpg")
    # print(index, name)
    outputs_merged = torch.squeeze(outputs_merged.detach().cpu())
    outputs_merged = outputs_merged.permute(1,2,0)
    outputs_merged = np.array(outputs_merged)
    return outputs_merged*255

opt.MODE =2
model = PasteModel(opt).to(opt.DEVICE)
model.load()
print(model)


cv2.setNumThreads(1)
torch.manual_seed(opt.SEED)
torch.cuda.manual_seed_all(opt.SEED)
np.random.seed(opt.SEED)
random.seed(opt.SEED)


if __name__ == '__main__':

    freeze_support()
    test()
