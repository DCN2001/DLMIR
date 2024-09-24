import os
import numpy as np
import librosa
import torch
import torch.utils.data

#Dataloader for trainset and validset
class NSynth(torch.utils.data.Dataset):
    def __init__(self,data_dir,mode,use_log):
        self.data_dir = os.path.join(data_dir,mode)        #The path of preprocessed NSynth dataset
        self.use_log = use_log                             #Use log or not (1 or 0)
        self.instru_list = os.listdir(self.data_dir)      #The instruments list of preprocessed NSynth dataset
        self.dataset = []       
        #dict mapping from instrument class to number (no inverse while training)
        self.label_map = {'bass': 0, 'brass': 1, 'flute': 2, 'guitar': 3, 'keyboard': 4, 'mallet': 5, 'organ': 6, 'reed': 7, 'string': 8, 'synth_lead': 9, 'vocal': 10}

        #Combine the data to the combination of [npy path & encoded_label]
        for instru in self.instru_list:
            audio_list = os.listdir(os.path.join(self.data_dir,instru))
            for audio in audio_list:
                label_encoding = [0] * len(self.label_map)  
                label_encoding[self.label_map[instru]] = 1      #Create one-hot vector
                self.dataset.append([os.path.join(self.data_dir,instru,audio), np.array(label_encoding, dtype=np.float32)])   #Datapath and Label
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_file = self.dataset[idx]      #The path of current data & label
        #Label
        label = audio_file[1]
        #Load the mel-spec
        mel_spec = np.load(audio_file[0])
        #If use log then turn log mel spectrogram
        if self.use_log:
            mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
        return mel_spec, label
    

#Function to call for dataloader when training 
def load_data(datapath, batch_size, use_log, num_workers):
    #Build dataset
    train_ds = NSynth(datapath, "train", use_log)
    valid_ds = NSynth(datapath, "valid", use_log)

    #Build dataloader
    trainset_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=num_workers)
    
    validset_loader = torch.utils.data.DataLoader(dataset=valid_ds,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=False,
                                                  drop_last=True,
                                                  num_workers=num_workers)
    
    return trainset_loader, validset_loader
    