import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"  # 该语句应当在出问题的语句前进行说明
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
import yaml
from model.edge_connect import EdgeConnect


class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.yaml = f.read()
            self.dict = yaml.safe_load(self.yaml)
            self.dict['PATH'] = os.path.dirname(config_path)   # 创建新的键值对

    # config取值时，如果有新输入的值则用新输入的值，否则采用默认值
    def __getattr__(self, name):
        if self.dict.get(name) is not None:  # 新设置值不为空，
            return self.dict[name]
        return None


def load_config(mode=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, choices=[1, 2, 3, 4],default=3,
                        help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')
    parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints/psv',
                        help='model checkpoints path (default: ./checkpoints)')

    if mode == 2 or mode == 3:
        # 可以选择用某个确切的文件用于测试
        # 为了得出对于测试集的不同指标 需要用整个数据集
        # 为了进行视觉上的比较，则需要对单张图片进行一一验证
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--masktype', type=int, default=4, help='3 for center and 4 for external')
        parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        parser.add_argument('--output', type=str, default="result2_", help='path to the output directory')
        parser.add_argument("--TEST_MASK_FLIST", type=str, default= r"E:\dataset\inpainting\mask\mask2.flist")

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')

    if not os.path.exists(args.path):
        os.makedirs(args.path)

    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    config = Config(config_path)

    if mode == 1:
        # model有四种，每种模型又有三种不同模式
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    elif mode == 2 or mode == 3:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3
        # config.INPUT_SIZE = 256
        if args.input is not None:
            config.TEST_FLIST = args.input
        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask
        if args.masktype is not None:
            config.MASK = args.masktype
        if args.edge is not None:
            config.TEST_EDGE_FLIST = args.edge
        if args.output is not None:
            config.RESULTS = os.path.join(config.PATH, args.output)
        if args.TEST_MASK_FLIST is not None:
            config.TEST_MASK_FLIST =args.TEST_MASK_FLIST


        config.MODE = mode
        config.MODEL = args.model if args.model is not None else 3
    return config


def main(mode=None):
    config = load_config(mode)

    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        config.DEVICE = torch.device("cpu")

    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # 加载模型，加载参数
    model = EdgeConnect(config)
    model.load()

    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    elif config.MODE == 3:
        print('\nstart testing...\n')
        # 设置一张图片 作为验证的标记
        model.test_as(1)

    else:
        print('\nstart eval...\n')
        model.eval()


if __name__ == "__main__":
    main()
