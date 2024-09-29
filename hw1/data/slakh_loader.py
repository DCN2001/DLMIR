import os
import json
import numpy as np
import librosa
import torch
import torch.utils.data


#Dataloader for Slakh2100 dataset in this homework
class Slakh(torch.utils.data.Dataset):
    def __init__(self,data_dir,mode):
        self.data_dir = os.path.join(data_dir,mode)                    #Path of .npy
        self.label_dir = os.path.join(data_dir,mode+"_labels.json")    #Path of metadata
        self.track_list = os.listdir(self.data_dir)                    #List all the track in the path
        self.dataset = []

        #Combine the data to [npy path, multi-hot label] and form the datasets
        with open(self.label_dir, 'r') as file:
            label_dict = json.load(file)
            for track in self.track_list:
                self.dataset.append([os.path.join(self.data_dir,track), np.array(label_dict.get(track), dtype=float)])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        #Load data 
        track_path = self.dataset[idx][0]
        wave = np.load(track_path)
        #Load label
        label = self.dataset[idx][1]
        return wave, label
    
#For training
def load_data(datapath, batch_size, num_workers):
    #Build dataset
    train_ds = Slakh(datapath, "train")
    valid_ds = Slakh(datapath, "validation")
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
    