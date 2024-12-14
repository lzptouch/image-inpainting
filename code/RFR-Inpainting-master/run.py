import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from model import RFRNetModel
from dataset import Dataset
from torch.utils.data import DataLoader

def run():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_root',default="../dataset1/celeba_train.flist", type=str)
    # parser.add_argument('--mask_root', default="../datasets/masktrain.flist",type=str)

    parser.add_argument('--data_root', default= r"E:\dataset\inpainting\places2\test.flist", type=str)
    parser.add_argument('--mask_root', default= r"E:\dataset\inpainting\mask\mask1.flist",type=str)

    parser.add_argument('--model_save_path', type=str, default='checkpoints/celeba')
    parser.add_argument('--result_save_path', type=str, default='checkpoints/places2/result1')
    parser.add_argument('--target_size', type=int, default=256)
    parser.add_argument('--mask_mode', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=450000)
    parser.add_argument('--model_path', type=str, default="checkpoints/places2/checkpoint_places2.pth")

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_threads', type=int, default=1)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--test', default=True,action='store_true')
    # parser.add_argument('--test',  action='store_true')
    # parser.add_argument('--gpu_id', type=str, default="0")
    args = parser.parse_args()
        
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    model = RFRNetModel()
    if args.test:
        model.initialize_model(args.model_path, False)
        model.cuda()
        data = Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse=True, training=False)
        dataloader = DataLoader(data)
        model.test(data,dataloader, args.result_save_path)
    else:
        model.initialize_model(args.model_path, True)
        model.cuda()
        data = Dataset(args.data_root, args.mask_root, args.mask_mode, args.target_size, mask_reverse = True)
        dataloader = DataLoader(data, batch_size = args.batch_size, shuffle = True, num_workers = args.n_threads)
        model.train(dataloader, args.model_save_path, args.finetune, args.num_iters)

if __name__ == '__main__':
    run()