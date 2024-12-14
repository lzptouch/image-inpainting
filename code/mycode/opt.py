import os
import torch
SEED= 2333
TEST_FLIST= "E:/dataset/inpainting/paris_/vali.flist"
TEST_MASK_FLIST= r"./mask.flist"
#
PATH = r'weight'

LR= 0.0001
D2G_LR= 0.1
BETA1= 0.0
BETA2= 0.9
INPUT_SIZE= 256
GAN_LOSS= "nsgan"

DIS_LOSS_WEIGHT= 1
GAN_LOSS_WEIGHT= 1

RESTRUCTION_LOSS_WEIGHT= 1
AUX_LOSS_WEIGHT = 0.1
STYLE_LOSS_WEIGHT= 250
CONTENT_LOSS_WEIGHT= 0.1
ADV_LOSS_WEIGHT= 0.1

iterat_interval=10000
max_iteration=10000000
if torch.cuda.is_available():
    print("可以使用cuda")
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # cudnn auto-tuner
else:
    DEVICE = torch.device("cpu")
