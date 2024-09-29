import os
import random
import torch
import torchaudio
import torch.utils.data
import torchaudio.transforms as T
import numpy as np
from tqdm import tqdm

class Trainset(torch.utils.data.Dataset):
    def __init__(self, trainset_dir):
        self.datapath = trainset_dir
        self.song_list = os.listdir(os.path.join(self.datapath,"mixture"))
        self.frame_seg = 15
        self.sample_per_track = 64
        self.samples = []       #One time
        
        for song in self.song_list:
            for i in range(self.sample_per_track):
                self.samples.append([os.path.join(self.datapath,"mixture",song),os.path.join(self.datapath,"vocal",song)])       #Mixture and Vocal(input and GT)
        
        self.trainset = self.samples + self.samples       #Two time for augmentation

    def __len__(self):
        return len(self.trainset)

    def augment_gain(self, mixture, target):
        """Applies a random gain between `low` and `high`"""
        low = 0.25
        high = 1.25
        g = low + torch.rand(1) * (high - low)
        return mixture * g, target * g

    # def augment_channelswap(self, mixture, target):
    #     """Swap channels of stereo signals with a probability of p=0.5"""
    #     if mixture.shape[0] == 2 and torch.tensor(1.0).uniform_() < 0.5:
    #         return torch.flip(mixture, [0]), torch.flip(target, [0])
    #     else:
    #         return mixture, target
    
    def __getitem__(self,idx):
        input_path = self.trainset[idx][0]
        label_path = self.trainset[idx][1]
        #for the first half no random mixing
        if idx < len(self.trainset)//2:
            mixture_mag = np.load(input_path)
            vocal_mag = np.load(label_path)
            song_len = mixture_mag.shape[-1]
            start_pt = np.random.randint(0, song_len-self.frame_seg)
            end_pt = start_pt + self.frame_seg
            mixture_mag = torch.tensor(mixture_mag[:,:,start_pt:end_pt])
            vocal_mag = torch.tensor(vocal_mag[:,:,start_pt:end_pt])
            #Augementation
            mixture_mag, vocal_mag = self.augment_gain(mixture_mag, vocal_mag)
        
        #after the first half => random mixing
        else:
            #Choose a segment for vocal 
            vocal_mag = np.load(label_path)
            vocal_len = vocal_mag.shape[-1]
            start_v_pt = np.random.randint(0, vocal_len-self.frame_seg)
            end_v_pt = start_v_pt + self.frame_seg
            vocal_mag = torch.tensor(vocal_mag[:,:,start_v_pt:end_v_pt])     
            #Choose a accompanian and choose a random segment to mix with vocal
            else_path = random.choice(self.samples)[1]
            else_path = else_path.replace('/vocal/','/else/')
            else_mag = np.load(else_path)
            else_len = else_mag.shape[-1]
            start_a_pt = np.random.randint(0, else_len-self.frame_seg)
            end_a_pt = start_a_pt + self.frame_seg
            else_mag = torch.tensor(else_mag[:,:,start_a_pt:end_a_pt]) 
            mixture_mag = vocal_mag + else_mag
            #Augementation
            mixture_mag, vocal_mag = self.augment_gain(mixture_mag, vocal_mag)
        return mixture_mag, vocal_mag


class Validset(torch.utils.data.Dataset):
    def __init__(self, validset_dir):
        self.datapath = validset_dir
        self.song_list = os.listdir(os.path.join(self.datapath,"mixture"))
        self.frame_seg = 15
        self.sample_per_track = 20
        self.validset = []      

        for song in self.song_list:
            for sample_idx in range(-self.sample_per_track//2,self.sample_per_track//2):
                self.validset.append([os.path.join(self.datapath,"mixture",song),os.path.join(self.datapath,"vocal",song),sample_idx])

    def __len__(self):
        return len(self.validset)

    def __getitem__(self, idx):
        mix_path, vocal_path, seg_idx = self.validset[idx]
        mixture_mag, vocal_mag = np.load(mix_path), np.load(vocal_path)
        #Select segment
        track_length = mixture_mag.shape[-1]
        mid_pt = track_length // 2
        start_pt = mid_pt + seg_idx*15
        end_pt = start_pt + self.frame_seg
        mixture_mag = torch.tensor(mixture_mag[:,:,start_pt:end_pt])
        vocal_mag = torch.tensor(vocal_mag[:,:,start_pt:end_pt])
        return mixture_mag, vocal_mag
        


def load_data(datapath, batch_size, num_workers):
    #Build Dataset
    trainpath = os.path.join(datapath,"train")
    validpath = os.path.join(datapath,"valid")
    train_ds = Trainset(trainpath)
    valid_ds = Validset(validpath)

    #Build Dataloader
    trainset_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                                batch_size=batch_size,
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=True,
                                                num_workers=num_workers)
    
    testset_loader = torch.utils.data.DataLoader(dataset=valid_ds,
                                                batch_size=1,
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=True,
                                                num_workers=num_workers)
    return trainset_loader, testset_loader



'''
datapath = "/home/data1/dcn2001/MUSDBHQ_HW/train"
train_ds = Trainset(datapath)
trainset_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                                batch_size=1,
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=True,
                                                num_workers=8)

for idx, batch in enumerate(tqdm(trainset_loader)):
    pass
'''




