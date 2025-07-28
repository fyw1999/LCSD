import os
import random
import argparse
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(description='Train') 
    parser.add_argument('--dataset', default='Mall',
                        help='dataset name')
    parser.add_argument('--scene', default='scene_001', type=str, 
                        help='scene name, for Mall and UCSD, set this to scene_001')
    parser.add_argument('--resource-path', type=str, default=r'/data/fyw/dataset/crowdcount/weakly-supervised-scene-specific/resources/',
                        help='path for pedestrians info and background images')
    parser.add_argument('--scene-dataset', default='mall_800_1200',
                        help='real data path')
    parser.add_argument('--save-dir', default='save',
                        help='path for saving model, synthetic datasets and log')
    parser.add_argument('--uncertain_thre', type=float, default=0.7,
                        help='iterative training epoch interval')
    parser.add_argument('--beta', type=float, default=8,
                        help='iterative training epoch interval')
    parser.add_argument('--loss_weight', type=float, default=0.01,
                        help='semi_loss weight')
    parser.add_argument('--iterative-num', type=int, default=20,
                        help='iterative training epoch interval')
    parser.add_argument('--start-iter', type=int, default=4,
                        help='iterative training epoch interval')
    parser.add_argument('--train-num', type=int, default=160, 
                        help='the number for training')
    parser.add_argument('--batch-size', type=int, default=20,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    
    args = parser.parse_args()
    return args

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    import torch
    from utils.my_trainer import MyTrainer
    args.seed = 42
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    trainer = MyTrainer(args)
    trainer.setup()
    trainer.train()