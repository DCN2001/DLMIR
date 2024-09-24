import os
import random
import torch
import torchaudio
import torch.utils.data
import torchaudio.transforms as T
import scipy.io.wavfile
import numpy as np

from tqdm import tqdm

class MUSDBHQ(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode):
        self.mode = mode
        self.segment_length = 22050 * 10
        self.data_dir = os.path.join(data_dir,mode)
        self.song_list = os.listdir(self.data_dir)
        self.dataset= []
        self.to_DB = T.AmplitudeToDB(stype='magnitude', top_db=None)
        if mode=="train":
            self.sample_per_songs = 100
            for song in self.song_list:
                   for sample in range(self.sample_per_songs):
                       self.dataset.append(os.path.join(self.data_dir,song))
        elif mode=="test":
            self.sample_per_songs = 5
            for song in self.song_list:
                   for sample_idx in range(-self.sample_per_songs//2,self.sample_per_songs//2):
                       self.dataset.append([os.path.join(self.data_dir,song),sample_idx])
        
    def __len__(self):
        return len(self.song_list) * self.sample_per_songs
    
    def __getitem__(self,idx):
        if self.mode == "train":
            track_path = self.dataset[idx]      #Current track
            mixture_wave = np.load(os.path.join(track_path,"mixture.npy"))   #Load mixture 
            track_length = mixture_wave.shape[1]    #Track length
            #Random select segment in the track, and segment the mixture
            start_pt = np.random.randint(0, track_length-self.segment_length)
            end_pt = start_pt + self.segment_length
            mixture_wave = torch.tensor(mixture_wave[:,start_pt:end_pt])
            #Load ground truth
            vocal_wave = np.load(os.path.join(track_path,"vocals.npy"))
            accompanian_wave = np.load(os.path.join(track_path,"accompanians.npy"))
            vocal_wave = torch.tensor(vocal_wave[:,start_pt:end_pt])     #GT-1
            accompanian_wave = accompanian_wave[:,start_pt:end_pt]   #GT-2
            # Transform input to STFT-spec
            spec_left = T.Spectrogram(n_fft=2048, win_length=1024, hop_length=512)(mixture_wave[0])
            spec_right = T.Spectrogram(n_fft=2048, win_length=1024, hop_length=512)(mixture_wave[1])
            mixture_spec = torch.stack([torch.abs(spec_left), torch.abs(spec_right)], dim=0)    
            # spec_left = T.MelSpectrogram(n_fft=2048, win_length=1024, hop_length=512, n_mels=128)(mixture_wave[0])
            # spec_right = T.MelSpectrogram(n_fft=2048, win_length=1024, hop_length=512, n_mels=128)(mixture_wave[1])
            # mixture_spec = torch.stack([spec_left, spec_right], dim=0)

            # Ground truth (GT) STFT-spec
            gt_left = T.Spectrogram(n_fft=2048, win_length=1024, hop_length=512)(vocal_wave[0])
            gt_right = T.Spectrogram(n_fft=2048, win_length=1024, hop_length=512)(vocal_wave[1])
            vocal_spec = torch.stack([torch.abs(gt_left), torch.abs(gt_right)], dim=0)
            # gt_left = T.MelSpectrogram(n_fft=2048, win_length=1024, hop_length=512, n_mels=128)(vocal_wave[0])
            # gt_right = T.MelSpectrogram(n_fft=2048, win_length=1024, hop_length=512, n_mels=128)(vocal_wave[1])
            # vocal_spec = torch.stack([gt_left, gt_right], dim=0)
            
        elif self.mode == "test":
            track_path, seg_index = self.dataset[idx]
            mixture_wave = np.load(os.path.join(track_path, "mixture.npy"))   #Load mixture
            track_length = mixture_wave.shape[1]
            #Select segment by segment index
            mid_pt = track_length // 2
            start_pt = mid_pt + seg_index*5*22050     #Shift 5 sec 
            end_pt = start_pt + self.segment_length
            mixture_wave = torch.tensor(mixture_wave[:,start_pt:end_pt])
            #Load ground truth 
            vocal_wave = np.load(os.path.join(track_path, "vocals.npy"))
            accompanian_wave = np.load(os.path.join(track_path, "accompanians.npy"))
            vocal_wave = torch.tensor(vocal_wave[:,start_pt:end_pt])
            accompanian_wave = accompanian_wave[:,start_pt:end_pt]
            # Transform input to STFT-spec
            spec_left = T.Spectrogram(n_fft=2048, win_length=1024, hop_length=512)(mixture_wave[0])
            spec_right = T.Spectrogram(n_fft=2048, win_length=1024, hop_length=512)(mixture_wave[1])
            mixture_spec = torch.stack([torch.abs(spec_left), torch.abs(spec_right)], dim=0)   
            #spec_left = T.MelSpectrogram(n_fft=2048, win_length=1024, hop_length=512, n_mels=128)(mixture_wave[0])
            #spec_right = T.MelSpectrogram(n_fft=2048, win_length=1024, hop_length=512, n_mels=128)(mixture_wave[1])
            #mixture_spec = torch.stack([spec_left, spec_right], dim=0)

            # Ground truth (GT) STFT-spec
            gt_left = T.Spectrogram(n_fft=2048, win_length=1024, hop_length=512)(vocal_wave[0])
            gt_right = T.Spectrogram(n_fft=2048, win_length=1024, hop_length=512)(vocal_wave[1])
            vocal_spec = torch.stack([torch.abs(gt_left), torch.abs(gt_right)], dim=0)
            #gt_left = T.MelSpectrogram(n_fft=2048, win_length=1024, hop_length=512, n_mels=128)(vocal_wave[0])
            #gt_right = T.MelSpectrogram(n_fft=2048, win_length=1024, hop_length=512, n_mels=128)(vocal_wave[1])
            #vocal_spec = torch.stack([gt_left, gt_right], dim=0)
        
        #Turn spectrogram to DB
        
        return mixture_spec, vocal_spec, vocal_wave
        

def load_data(datapath, batch_size, num_workers):
    #Build Dataset
    train_ds = MUSDBHQ(datapath, "train")
    test_ds = MUSDBHQ(datapath, "test")

    #Build Dataloader
    trainset_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                                batch_size=batch_size,
                                                pin_memory=True,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=num_workers)
    
    testset_loader = torch.utils.data.DataLoader(dataset=test_ds,
                                                batch_size=batch_size,
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=True,
                                                num_workers=num_workers)
    return trainset_loader, testset_loader

                                          



datapath = "/home/data1/dcn2001/MUSDBHQ_2"
train_ds = MUSDBHQ(datapath, "train")
trainset_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                                batch_size=1,
                                                pin_memory=True,
                                                 shuffle=False,
                                                drop_last=True,
                                                  num_workers=1)

for idx, batch in enumerate(tqdm(trainset_loader)):
     pass