import sys, os
abs_path = (os.path.abspath(__file__)).split("/")[:-2]
abs_path += ["common_utils"]
sys.path.append("/".join(abs_path))
print(sys.path)

import argparse
import torch
from utils import *
from networks import AudioVisualModel, AudioOnlyModel, test_model
from solver import Solver
import warnings
import random
warnings.filterwarnings("ignore", category=UserWarning) 

seed = 2023

def main(args):
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # seeding
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # speaker id assignment
    mix_lst=open(args.mix_lst_path).read().splitlines()
    train_lst=list(filter(lambda x: x.split(',')[0]=='train', mix_lst))
    IDs = 0
    speaker_dict={}
    for line in train_lst:
        for i in range(2):
            ID = line.split(',')[i*4+2]
            if ID not in speaker_dict:
                speaker_dict[ID]=IDs
                IDs += 1
    args.speaker_dict=speaker_dict
    args.speakers=len(speaker_dict)

    # Model
    # model = AudioOnlyModel()
    model = AudioVisualModel()
    # print(model)
    if (args.distributed and args.local_rank ==0) or args.distributed == False:
        print("speaker_dict length:", len(speaker_dict))
        print("started on " + args.log_name + '\n')
        # print(args)
        print("\nTotal number of parameters: {} Mb\n".format(sum(p.numel() for p in model.parameters())/ 10**6))
        # print(model)

    model = model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    train_sampler, train_generator = get_dataloader(args, 'train')
    _, val_generator = get_dataloader(args, 'val')
    _, test_generator = get_dataloader(args, 'test')
    args.train_sampler = train_sampler

    solver = Solver(args = args,
                model = model,
                optimizer = optimizer,
                train_data = train_generator,
                validation_data = val_generator,
                test_data=test_generator) 
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("av-yolo")
    
    # Dataloader
    parser.add_argument('--mix_lst_path', type=str, default='/workspace/liuqinghua/code/masking_ss/data_preprocessing/lrs3/mixture_20k.csv')
    parser.add_argument('--mixture_direc', type=str, default='/workspace/liuqinghua/datasets/lrs3/wav/mixture/')
    parser.add_argument('--audio_direc', type=str, default='/workspace/liuqinghua/datasets/lrs3/wav/')
    parser.add_argument('--video_direc', type=str, default='/workspace/liuqinghua/datasets/lrs3/')
    parser.add_argument('--max_length', default=6, type=int)
    parser.add_argument('--mask_percentage', default=0.2, type=float,
                    help='Percentage of masking video sequence')
    
    # Training    
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of workers to generate minibatch')
    parser.add_argument('--epochs', default=30, type=int,
                        help='Number of maximum epochs')
    parser.add_argument('--mask_type', default='repeat', type=str, help='gaussian, speckle, substitution, repeat, and other for unmasked')

    # Model hyperparameters
    parser.add_argument('--L', default=40, type=int,
                        help='Length of the filters in samples (80=5ms at 16kHZ)')
    parser.add_argument('--N', default=256, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--C', type=int, default=2,
                        help='number of speakers to mix')
    parser.add_argument('--pretrain_grad', default=False, type=str2bool,
                        help='Set up the parameter of pretained model')
    parser.add_argument('--feature_layers', default=[None], type=int, nargs='+',
                        help='List of feature layers')

    # optimizer
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Init learning rate')
    parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')


    # Log and Visulization
    parser.add_argument('--log_name', type=str, default='test',
                        help='the name of the log')
    parser.add_argument('--use_tensorboard', type=int, default=0,
                        help='Whether to use use_tensorboard')
    parser.add_argument('--continue_from', type=str, default='',
                        help='Whether to resume training')

    # Distributed training
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])

    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        args.world_size = int(os.environ['WORLD_SIZE'])

    print('Experiment Config:\n', args)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
    
    # main(args)
    test_model()    
    

