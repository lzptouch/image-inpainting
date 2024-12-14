import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # 该语句应当在出问题的语句前进行说明
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.misf import MISF
import torch.nn as nn


def main(mode=None):
    config = load_config(mode)
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")
    cv2.setNumThreads(0)


    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)


    model = MISF(config)
    model.load()

    iteration = model.inpaint_model.iteration
    if len(config.GPU) > 1:
        print('GPU:{}'.format(config.GPU))
        model.inpaint_model.generator = nn.DataParallel(model.inpaint_model.generator, config.GPU)
        model.inpaint_model.discriminator = nn.DataParallel(model.inpaint_model.discriminator, config.GPU)

    model.inpaint_model.iteration = iteration

    # print(model.inpaint_model)



    # model training
    if config.MODE == 1:
        # config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()

def load_config(mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints/paris',
                        help='model checkpoints path (default: ./checkpoints)')
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4], default=3, help='')
    parser.add_argument('--output', type=str, default="result1", help='path to the output directory')
    parser.add_argument("--TEST_MASK_FLIST", type=str, default=r"F:\datasets\inpainting\mask\mask1.flist")

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    config = Config(config_path)
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    elif mode == 2:
        config.MODE = 2
        if args.output is not None:
            config.RESULTS = os.path.join(config.PATH, args.output)
        if args.TEST_MASK_FLIST is not None:
            config.TEST_MASK_FLIST =args.TEST_MASK_FLIST

    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3
    return config


if __name__ == "__main__":
    main()
