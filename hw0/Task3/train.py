from model.SC_CNN import ShortChunkCNN_Res
from model.CNNSA import CNNSA
from config import get_config
import data.nsynth_loader as dataloader

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm 


#The main pipeline of training
class Trainer():
    def __init__(self, train_loader, valid_loader, gpu_id: int):
        self.device = torch.device(gpu_id)
        #Dataloader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        #Select model
        if args.model == "SC_CNN":
            self.model = ShortChunkCNN_Res(n_channels=128,n_class=11).to(self.device)
            summary(self.model, (1 , 128, 251))
        elif args.model == "CNNSA":
            self.model = CNNSA(n_channels=128, n_class=11).to(self.device)
            summary(self.model, (1, 128, 251))
        
        #Define optimizer and scheduler (schedule rule: half the lr if valid loss didn'nt decrease for two epoch)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr, weight_decay=args.l2_lambda)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=args.decay_epoch, factor=0.5)
    
    def criterion(self, predict, label):
        #Define loss function
        CE_loss = nn.CrossEntropyLoss()
        #Estimate total loss
        total_loss = CE_loss(predict,label)
        return total_loss

    @torch.no_grad()
    def valid_batch(self,batch):
        input = batch[0].to(self.device)
        label = batch[1].to(self.device)
        predict = self.model(input)
        
        #Verify the answer
        predict_class = torch.argmax(predict,dim=1)
        predict_3class = torch.topk(predict, k=3, dim=1).indices
        answers = torch.argmax(label, dim=1)
        #Estimating acc and loss
        top1_acc_batch = (predict_class==answers).float().mean()
        top3_acc_batch = predict_3class.eq(answers.view(-1, 1)).sum(dim=1).float().mean()
        loss_batch = self.criterion(predict, label)
        return loss_batch.item(), top1_acc_batch.item(), top3_acc_batch.item()
    
    def valid_total(self):
        top1_acc_total = 0.0
        top3_acc_total = 0.0
        loss_total = 0.0
        for idx, batch in enumerate(tqdm(self.valid_loader, desc="Eval bar", colour="#9F35FF")):
            step = idx + 1
            loss_batch, top1_acc_batch, top3_acc_batch = self.valid_batch(batch)        #Call for validating a batch
            #Accumalting acc and loss
            top1_acc_total += top1_acc_batch
            top3_acc_total += top3_acc_batch
            loss_total += loss_batch

        #Total loss & acc for the whole validation set
        top1_acc_total = top1_acc_total/step
        top3_acc_total = top3_acc_total/step
        loss_total = loss_total/step
        return loss_total, top1_acc_total, top3_acc_total


    def train_batch(self, batch):
        input = batch[0].to(self.device)
        label = batch[1].to(self.device)
        predict = self.model(input)             #Forward propogation
        loss = self.criterion(predict, label)   #Estimate train loss
        self.optimizer.zero_grad()              #Clear the gradient in optimizer
        loss.backward()                         #Backward propogation
        self.optimizer.step()                   #Optimize
        return loss.item()

        
    def train_total(self):
        train_loss_list = []
        valid_loss_list = []
        top1_acc_list = []
        top3_acc_list = []
        min_val_loss = np.inf   #Initialize the minimum valid loss
        for epoch in tqdm(range(args.epochs), desc="Epoch", colour="#0080FF"):
            self.model.train()  
            train_loss = 0.0
            for idx, batch in enumerate(tqdm(self.train_loader, desc=f"Train bar({epoch})", colour="#ff7300")):
                step = idx + 1
                loss_batch = self.train_batch(batch)        #Call for training a batch
                train_loss += loss_batch
            train_loss_list.append(train_loss/step)
            print(f"\n train loss: {train_loss/step}")

            self.model.eval()
            valid_loss, valid_top1_acc, valid_top3_acc = self.valid_total()     #Validate every epoch after training
            top1_acc_list.append(valid_top1_acc)        
            top3_acc_list.append(valid_top3_acc)
            valid_loss_list.append(valid_loss)
            print(f"\n valid loss: {valid_loss} | Top1 acc: {valid_top1_acc} | Top3 acc: {valid_top3_acc}")     #Show the valid loss and acc
            #If the valid loss is the best, then save model check point
            if valid_loss<min_val_loss:
                print(f"Saving model at epoch: {epoch}")
                min_val_loss = valid_loss
                torch.save(self.model.state_dict(), args.model_save_path)
            self.scheduler.step(valid_loss)
            
        #Draw curve for acc and loss versus epoch
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(top1_acc_list)), top1_acc_list, label='Top1 acc', color='orange')
        plt.plot(range(len(top3_acc_list)), top3_acc_list, label='Top3 acc', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.title(f'Acc vs. Epoch best: {np.argmax(top1_acc_list)} | {np.max(top1_acc_list)}')
        plt.legend()
        plt.grid(True)
        plt.savefig(args.curve_save_path+'/acc.png')  
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss', color='blue')
        plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid_loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Best ckpt: {np.argmin(valid_loss_list)} | {np.min(valid_loss_list)}')
        plt.legend()
        plt.grid(True)
        plt.savefig(args.curve_save_path+'/loss.png')   
        plt.close()



#-------------------------------------------------------------------------------------
def main(args):
    # Create dataloader
    train_loader, valid_loader = dataloader.load_data(args.data_path, args.batch_size, args.use_log, 8)   
    #Set cuda
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
    trainer = Trainer(train_loader, valid_loader, gpu_id=gpu_id)
    trainer.train_total()

if __name__ == "__main__":
    args = get_config()     #Config from config.py 
    main(args)