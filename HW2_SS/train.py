from model.deep_unet import UNet
from model.open_unmix import OpenUnmix
from model.dae import DAESkipConnections
import data.MUSDBHQ_aug_loader as dataloader
from config import get_config

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import mir_eval
import torch.nn.functional as F
import torchaudio
from torchinfo import summary
import torchaudio.transforms as T
from torch.nn.utils import clip_grad_value_
import librosa
from tqdm import tqdm


args = get_config()
class Trainer():
    def __init__(self, train_loader, valid_loader, gpu_id: int):
        self.gpu_id = gpu_id
        self.device = torch.device(f'cuda:{self.gpu_id}' if torch.cuda.is_available() else 'cpu')
        #Dataloader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = OpenUnmix(nb_bins = 1025,nb_channels = 1,hidden_size = 512,nb_layers = 6).to(self.device)
        #self.model = UNet(in_channels=1).to(self.device)
        #self.model = DAESkipConnections().to(self.device)
        summary(self.model, (1 , 1, 1025, 15))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr, weight_decay=args.l2_lambda)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=2, factor=0.5)
        
    
    def est_loss(self, vocal_out, vocal_label):
        #Define loss
        MSE_loss = torch.nn.MSELoss()
        # Compute loss
        total_loss = MSE_loss(vocal_out, vocal_label)
        return total_loss
    
    def est_metric(self, vocal_out, vocal_label):
        batch_sdr = 0.0
        vocal_out_np = vocal_out.cpu().numpy()
        vocal_label_np = vocal_label.cpu().numpy()
        count = 0
        for i in range(vocal_out.shape[0]):
            count += 1
            if not np.all(vocal_label_np[i,0,:] == 0):
                sdr_left, _, _, _ = mir_eval.separation.bss_eval_sources(vocal_label_np[i,0,:], vocal_out_np[i,0,:])
                sdr_right, _, _, _ = mir_eval.separation.bss_eval_sources(vocal_label_np[i,1,:], vocal_out_np[i,1,:])
                sdr = (sdr_left + sdr_right) / 2
                batch_sdr += sdr[0] 
            else:
                pass
        return batch_sdr/count

    @torch.no_grad()
    def valid_batch(self,batch):
        input = batch[0].to(self.device)
        vocal_label = batch[1].to(self.device)
        vocal_out = self.model(input)
        loss_batch = self.est_loss(vocal_out,vocal_label)
        return loss_batch.item()
    
    def valid_total(self):
        loss_total = 0.0
        for idx, batch in enumerate(tqdm(self.valid_loader, desc="Eval bar", colour="#9F35FF")):
            step = idx + 1
            loss_batch = self.valid_batch(batch)
            loss_total += loss_batch
        loss_total = loss_total/step
        return loss_total


    def train_batch(self, batch):
        input = batch[0].to(self.device)
        vocal_label = batch[1].to(self.device)
        vocal_out = self.model(input)
        loss = self.est_loss(vocal_out, vocal_label)
        self.optimizer.zero_grad()
        loss.backward()
        #clip_grad_value_(self.model.parameters(), clip_value=0.5) 
        self.optimizer.step()
        return loss.item()

        
    def train_total(self):
        train_loss_list = []
        valid_loss_list = []
        min_valid_loss = np.Inf
        for epoch in tqdm(range(args.epochs), desc="Epoch", colour="#0080FF"):
            self.model.train()
            train_loss = 0.0
            for idx, batch in enumerate(tqdm(self.train_loader, desc=f"Train bar({epoch})", colour="#ff7300")):
               step = idx + 1
               loss_batch = self.train_batch(batch)
               train_loss += loss_batch
            train_loss_list.append(train_loss/step)

            self.model.eval()
            valid_loss = self.valid_total()
            valid_loss_list.append(valid_loss)
            print(f"\n train loss: {train_loss/step} | valid loss: {valid_loss}")
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                saved_model_path = args.model_save_path
                print(f"Saving model epoch: {epoch}................")
                torch.save(self.model.state_dict(), saved_model_path)
            self.scheduler.step(valid_loss)
        
        #Draw curve
        plt.figure(figsize=(12, 6)) 
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss', color='blue')
        plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid_loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss vs. Epoch best')
        plt.legend()
        plt.grid(True)
        plt.savefig(args.curve_save_path)  
        plt.close()



#-------------------------------------------------------------------------------------
def main(args):
    # Create dataloader
    train_loader, test_loader = dataloader.load_data(args.data_path, args.batch_size, 8)   
    trainer = Trainer(train_loader, test_loader, gpu_id=3)
    trainer.train_total()

if __name__ == "__main__":
    main(args)