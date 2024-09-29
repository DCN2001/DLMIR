import os
import numpy as np
import torch
import torch.nn
import torchaudio
from tqdm import tqdm

valid_song = [  'Actions - One Minute Smile','Clara Berry And Wooldog - Waltz For My Victims',
                'Johnny Lokke - Promises & Lies','Patrick Talbot - A Reason To Leave',
                'Triviul - Angelsaint','Alexander Ross - Goodbye Bolero',
                'Fergessen - Nos Palpitants','Leaf - Summerghost',
                'Skelpolu - Human Mistakes','Young Griffo - Pennies',
                'ANiMAL - Rockshow','James May - On The Line',
                'Meaxic - Take A Step','Traffic Experiment - Sirens']



def preprocess_train(org_path,new_path,mode):
    n_fft, win_length, hop_length = 2048, 2048, 512
    resampler = torchaudio.transforms.Resample(44100, 16000)
    stft = torchaudio.transforms.Spectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length,power=None)
    track_list = os.listdir(org_path)
    if mode=="train":
        for track in tqdm(track_list):
            if track not in valid_song:
                #Load wave 
                track_folder = os.path.join(org_path,track)
                mix_wav, sr = torchaudio.load(os.path.join(track_folder,"mixture.wav"))   #Load
                vocal_wav, sr = torchaudio.load(os.path.join(track_folder,"vocals.wav"))
                bass_wav, sr = torchaudio.load(os.path.join(track_folder,"bass.wav"))
                drum_wav, sr = torchaudio.load(os.path.join(track_folder,"drums.wav"))
                other_wav, sr =  torchaudio.load(os.path.join(track_folder,"other.wav"))
                else_wav = bass_wav + drum_wav + other_wav          #Linear add don't need to average I am sure
                #Resample
                mix_wav, vocal_wav, else_wav = resampler(mix_wav), resampler(vocal_wav), resampler(else_wav)
                #Transform to mono channel
                mix_wav, vocal_wav, else_wav = torch.mean(mix_wav,0,keepdim=True), torch.mean(vocal_wav,0,keepdim=True), torch.mean(else_wav,0,keepdim=True)
                #Padding
                n_sample_points = mix_wav.shape[1]
                pad_size = (max(0, n_sample_points - win_length) // hop_length + 1) * hop_length - n_sample_points + win_length
                mix_wav = torch.nn.functional.pad(mix_wav,(0, pad_size),mode='constant',value=0.0)
                vocal_wav = torch.nn.functional.pad(vocal_wav,(0, pad_size),mode='constant',value=0.0)
                else_wav = torch.nn.functional.pad(else_wav,(0, pad_size),mode='constant',value=0.0)
                #STFT
                mix_spec, vocal_spec, else_spec = stft(mix_wav), stft(vocal_wav), stft(else_wav)
                #Take magnitude
                mix_mag, vocal_mag, else_mag = torch.abs(mix_spec), torch.abs(vocal_spec), torch.abs(else_spec)
                #Save to .pt
                np.save(os.path.join(new_path,"mixture",track+".npy"),mix_mag.numpy())
                np.save(os.path.join(new_path,"vocal",track+".npy"),vocal_mag.numpy())
                np.save(os.path.join(new_path,"else",track+".npy"),else_mag.numpy())
                # torch.save(mix_mag, os.path.join(new_path,"mixture",track+".pt"))
                # torch.save(vocal_mag, os.path.join(new_path,"vocal",track+".pt"))
                # torch.save(else_mag, os.path.join(new_path,"else",track+".pt"))
    
    elif mode=="valid":
        for track in tqdm(track_list):
            if track in valid_song:
                #Load wave 
                track_folder = os.path.join(org_path,track)
                mix_wav, sr = torchaudio.load(os.path.join(track_folder,"mixture.wav"))   #Load
                vocal_wav, sr = torchaudio.load(os.path.join(track_folder,"vocals.wav"))
                #Resample
                mix_wav, vocal_wav = resampler(mix_wav), resampler(vocal_wav)
                #Transform to mono channel
                mix_wav, vocal_wav = torch.mean(mix_wav,0,keepdim=True), torch.mean(vocal_wav,0,keepdim=True)
                #Padding
                n_sample_points = mix_wav.shape[1]
                pad_size = (max(0, n_sample_points - win_length) // hop_length + 1) * hop_length - n_sample_points + win_length
                mix_wav = torch.nn.functional.pad(mix_wav,(0, pad_size),mode='constant',value=0.0)
                vocal_wav = torch.nn.functional.pad(vocal_wav,(0, pad_size),mode='constant',value=0.0)
                #STFT
                mix_spec, vocal_spec = stft(mix_wav), stft(vocal_wav)
                #Take magnitude
                mix_mag, vocal_mag = torch.abs(mix_spec), torch.abs(vocal_spec)
                #Save to .pt
                np.save(os.path.join(new_path,"mixture",track+".npy"),mix_mag.numpy())
                np.save(os.path.join(new_path,"vocal",track+".npy"),vocal_mag.numpy())
                # torch.save(mix_mag, os.path.join(new_path,"mixture",track+".pt"))
                # torch.save(vocal_mag, os.path.join(new_path,"vocal",track+".pt"))
    





if __name__ == "__main__":
    preprocess_train("/home/data1/dcn2001/MUSDBHQ/train","/home/data1/dcn2001/MUSDBHQ_HW/train","train")
    preprocess_train("/home/data1/dcn2001/MUSDBHQ/train","/home/data1/dcn2001/MUSDBHQ_HW/valid","valid")