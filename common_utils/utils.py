import numpy as np
import random
import math
import torch.distributed as dist
import torch
import torch.nn as nn
import torch.utils.data as data
import scipy.io.wavfile as wavfile
from itertools import permutations
import tqdm
import os
import cv2
from argparse import ArgumentTypeError

EPS = 1e-6

class dataset(data.Dataset):
    def __init__(self,
                speaker_dict,
                mix_lst_path,
                audio_direc,
                video_direc,
                mixture_direc,
                mask_type,
                batch_size,
                mask_percentage=0.2,
                partition='test',
                audio_only=False,
                sampling_rate=16000,
                max_length=3,
                mix_no=2):

        self.minibatch =[]
        self.audio_only = audio_only
        self.audio_direc = audio_direc
        self.video_direc = video_direc
        self.mixture_direc = mixture_direc
        self.sampling_rate = sampling_rate
        self.partition = partition
        self.max_length = max_length
        self.C=mix_no
        self.speaker_id=speaker_dict
        self.mask_type = mask_type
        self.mask_percentage=mask_percentage

        mix_lst=open(mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))
        
        assert (batch_size%self.C) == 0, "input batch_size should be multiples of mixture speakers"

        self.batch_size = int(batch_size/self.C )
        sorted_mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
        start = 0
        while True:
            end = min(len(sorted_mix_lst), start + self.batch_size)
            self.minibatch.append(sorted_mix_lst[start:end])
            if end == len(sorted_mix_lst):
                break
            start = end

    def __getitem__(self, index):
        batch_lst = self.minibatch[index]
        min_length = int(float(batch_lst[-1].split(',')[-1])*self.sampling_rate)
        visual_min_length = math.floor(min_length/self.sampling_rate*25)
        
        mixtures=[]
        audios=[]
        visuals=[]
        speakers=[]
        for line in batch_lst:
            mixture_path=self.mixture_direc+self.partition+'/'+ line.replace(',','_').replace('/','_')+'.wav'
            _, mixture = wavfile.read(mixture_path)
            mixture = self._audio_norm(mixture[:min_length])

            
            line=line.split(',')
            for c in range(self.C):
                # read target audio
                audio_path=self.audio_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.wav'
                _, audio = wavfile.read(audio_path)
                audios.append(self._audio_norm(audio[:min_length]))

                # read target audio id
                if self.partition == 'test':
                    speakers.append(0)
                else: speakers.append(self.speaker_id[line[c*4+2]])

                # read target visual reference
                video_path=self.video_direc+line[c*4+1]+'/'+line[c*4+2]+'/'+line[c*4+3]+'.mp4'
                captureObj = cv2.VideoCapture(video_path)
                roiSequence = []
                roiSize = 112
                while (captureObj.isOpened()):
                    ret, frame = captureObj.read()
                    if ret == True:
                        if self.mask_type == 'gaussian':
                            frame = np.asarray(frame/255, dtype=np.float32)
                            noise = np.random.normal(0.1, 0.1, frame.shape).astype(dtype=np.float32)
                            frame = frame + noise                           
                            frame = np.clip(frame, 0, 1)
                            frame = np.uint8(frame*255)
                        elif self.mask_type == 'speckle':
                            h,w,c = frame.shape
                            frame = np.asarray(frame/255, dtype=np.float32)                        
                            noise = np.random.randn(h,w,c)
                            frame = frame + frame*noise
                            frame = np.clip(frame, 0, 1)
                            frame = np.uint8(frame*255)
                        # cv2.imwrite("frame.jpg", frame)
                        
                        grayed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        grayed = grayed/225
                        grayed = cv2.resize(grayed, (roiSize * 2, roiSize * 2))
                        roi = grayed[int(roiSize - (roiSize / 2)):int(roiSize + (roiSize / 2)),
                            int(roiSize - (roiSize / 2)):int(roiSize + (roiSize / 2))]
                        roiSequence.append(roi)
                    else:
                        break
                captureObj.release()

                roiSequence_cut = roiSequence[:visual_min_length]
                if self.mask_type == 'substitution':
                    seq_len = min(visual_min_length, self.max_length*25)
                    mask_length = int(min(random.random()*seq_len, self.mask_percentage*seq_len))
                    mask_start = np.random.randint(0, seq_len-mask_length)
                    # print("mask_start:{} mask_length:{}".format(mask_start, mask_length))
                    sub_list = roiSequence[-mask_length:]
                    roiSequence_cut[mask_start:mask_start+mask_length] = sub_list
                elif self.mask_type == 'repeat':
                    seq_len = min(visual_min_length, self.max_length*25)
                    mask_length = int(min(random.random()*seq_len, self.mask_percentage*seq_len))
                    mask_start = np.random.randint(0, seq_len-mask_length)
                    repeat_list = [roiSequence[mask_start] for i in range(mask_length)]
                    roiSequence_cut[mask_start:mask_start+mask_length] = repeat_list
                    # print("mask_start:{} mask_length:{} list_len:{}".format(mask_start, mask_length, len(repeat_list)))

                # cv2.imwrite("roiSequence.jpg", np.floor(225*np.concatenate(roiSequence_cut, axis=1)).astype(np.int32))
                visual = np.asarray(roiSequence_cut)    # num_frames, 112, 112                
                if visual.shape[0] < visual_min_length:
                    visual = np.pad(visual, ((0, int(visual_min_length - visual.shape[0])), (0,0), (0,0)), mode = 'edge')
                visuals.append(visual)

                # read overlapped speech
                mixtures.append(mixture)
        
        return np.asarray(mixtures)[:,:self.max_length*self.sampling_rate], \
                np.asarray(audios)[:,:self.max_length*self.sampling_rate], \
                np.asarray(visuals)[:,:self.max_length*25,...], \
                np.asarray(speakers)

    def __len__(self):
        return len(self.minibatch)

    def _audio_norm(self,audio):
        return np.divide(audio, np.max(np.abs(audio)))


class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()
            ind = torch.randperm(int(len(self.dataset)/self.num_replicas), generator=g)*self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind+i).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dataloader(args, partition):
    datasets = dataset(
                speaker_dict =args.speaker_dict,
                mix_lst_path=args.mix_lst_path,
                audio_direc=args.audio_direc,
                video_direc=args.video_direc,
                mixture_direc=args.mixture_direc,
                mask_type=args.mask_type,
                batch_size=args.batch_size,
                max_length=args.max_length,
                partition=partition,
                
                mix_no=args.C)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler)

    return sampler, generator


if __name__ == '__main__':
    mix_lst=open('/workspace/liuqinghua/datasets/voxceleb2/vox2_mixture_20k.csv').read().splitlines()
    train_lst=list(filter(lambda x: x.split(',')[0]=='train', mix_lst))
    IDs = 0
    speaker_dict={}
    for line in train_lst:
        for i in range(2):
            ID = line.split(',')[i*4+2]
            if ID not in speaker_dict:
                speaker_dict[ID]=IDs
                IDs += 1

    datasets = dataset(
                speaker_dict,
                mix_lst_path='/workspace/liuqinghua/datasets/lrs3/lrs3_mixture_20k.csv',
                audio_direc='/workspace/liuqinghua/datasets/lrs3/wav/',
                video_direc='/workspace/liuqinghua/datasets/lrs3/',
                mixture_direc='/workspace/liuqinghua/datasets/lrs3/wav/mixture/',
                mask_type='repeat',   # gaussian, speckle, substitution, repeat
                batch_size=2,
                partition='test')
    data_loader = data.DataLoader(datasets,
                batch_size = 1,
                shuffle= False,
                num_workers = 4)

    for a_mix, a_tgt, v_tgt, speakers in tqdm.tqdm(data_loader):
        print(a_mix.squeeze().size())
        print(a_tgt.squeeze().size())
        print(v_tgt.squeeze().size())
        break
        # pass

def str2bool(arg:str):
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError("unsupported bool value, please check argument(s) with boolean type")

    # a = np.ones((24,512))
    # print(a.shape)
    # a = np.pad(a, ((0,-1), (0,0)), 'edge')
    # print(a.shape)

    # a = np.random.rand(2,2,3)
    # print(a)
    # a = a.reshape(4,3)
    # print(a)