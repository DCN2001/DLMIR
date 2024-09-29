from model.MERT_Classifier import MERT_Classifier

from config import get_config
from data.slakh_loader import load_data, Slakh

import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm



#The main pipeline of training
class Trainer():
    def __init__(self, train_loader, valid_loader, gpu_id: int):
        self.device = torch.device(gpu_id)
        #Dataloader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        #Select model
        self.model = MERT_Classifier(n_class=9, freeze=args.freeze).to(self.device)
        summary(self.model, (1 , 120000))
        
        #Define optimizer and scheduler (schedule rule: half the lr if valid loss didn'nt decrease for two epoch)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr, weight_decay=args.l2_lambda)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=args.decay_epoch, factor=0.5)
    
    def criterion(self, predict, label):
        #Define loss function
        BCE_loss = nn.BCEWithLogitsLoss()   #Since the model output are logits
        #Estimate total loss
        total_loss = BCE_loss(predict,label)
        return total_loss

    @torch.no_grad()
    def valid_batch(self,batch):
        input = batch[0].to(self.device)
        label = batch[1].to(self.device)
        predict = self.model(input)        #Model output
        #Estimating loss
        loss_batch = self.criterion(predict, label)
        return loss_batch.item()
    
    def valid_total(self):
        loss_total = 0.0
        for idx, batch in enumerate(tqdm(self.valid_loader, desc="Eval bar", colour="#9F35FF")):
            step = idx + 1
            loss_batch = self.valid_batch(batch)        #Call for validating a batch
            #Accumalting loss
            loss_total += loss_batch

        #Total loss for the whole validation set
        loss_total = loss_total/step
        return loss_total

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
        min_val_loss = np.inf       #Initialize the minimum valid loss as infinity
        for epoch in tqdm(range(args.epochs), desc="Epoch", colour="#0080FF"):
            self.model.train()  
            train_loss = 0.0
            for idx, batch in enumerate(tqdm(self.train_loader, desc=f"Train bar({epoch})", colour="#ff7300")):
                step = idx + 1
                loss_batch = self.train_batch(batch)        #Call for training a batch
                train_loss += loss_batch
            train_loss_list.append(train_loss/step)

            self.model.eval()
            valid_loss = self.valid_total()     #Validate every epoch after training
            valid_loss_list.append(valid_loss)
            print(f"\n train loss: {train_loss/step} | valid loss: {valid_loss}")     #Show the valid loss and acc
            #If the valid loss is the best, then save model check point
            if valid_loss<min_val_loss:
                print(f"Saving model at epoch: {epoch}")
                min_val_loss = valid_loss
                torch.save(self.model.state_dict(), args.model_save_path)
            self.scheduler.step(valid_loss)
            
        #Draw curve for loss versus epoch
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
    


#The function for finding the optimal threshold of multi-label classification (By F1-score micro avg)
@torch.no_grad()
def find_threshold(args):
    #Build dataloader
    test_ds = Slakh(args.data_path,"test")
    test_loader = torch.utils.data.DataLoader(dataset=test_ds,
                                              batch_size=16,
                                              pin_memory=True,
                                              shuffle=False,
                                              drop_last=True,
                                              num_workers=8)
    
    #Build model
    classifier = MERT_Classifier(n_class=9,freeze=False).to('cuda')
    classifier.load_state_dict(torch.load(args.model_save_path, weights_only=True))
    classifier.eval()

    #Grid search for threshold
    thrs = np.arange(0.0, 1.0, 0.05)
    best_f1 = 0.0
    best_thr = 0.0

    #Start searching the optimal threshold
    for thr in tqdm(thrs, desc="Threshold search", colour="red"):
        predict_list = []
        label_list = []
        for idx, batch in enumerate(tqdm(test_loader, desc="Testing", colour="blue")):
            #Load input and label
            input = batch[0].to('cuda')
            label = batch[1]

            #Feed input to model
            outputs = classifier(input)
            probs = torch.sigmoid(outputs).cpu().numpy()

            #Convert to 0 & 1
            predict = (probs >= thr).astype(int)

            #Append to list
            predict_list.append(predict)
            label_list.append(label)
        
        #Flatten
        predicts = np.concatenate(predict_list,axis=0)
        labels = np.concatenate(label_list, axis=0)
        
        #Estimate f1 score
        f1 = f1_score(labels,predicts, average='micro')

        #Check if the best
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    #Final result of the current model
    print(f"Best threshold: {best_thr} | Best F1 score: {best_f1}")



#-------------------------------------------------------------------------------------
def main(args):
    # Create dataloader
    train_loader, valid_loader = load_data(args.data_path, args.batch_size, args.num_workers)   
    #Set cuda
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
    trainer = Trainer(train_loader, valid_loader, gpu_id=gpu_id)
    trainer.train_total()
    #Find threshold for the current model by testset
    find_threshold(args)

if __name__ == "__main__":
    args = get_config()     #Configuration from config.py 
    main(args)