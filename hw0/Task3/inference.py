from model.SC_CNN import ShortChunkCNN_Res
from model.CNNSA import CNNSA
import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

#Class for evaluating on orginal NSynth testset
class Classifier():
    def __init__(self, args, gpu_id: int):
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')    #Set cuda device
        self.testset_path = args.testset_path           #Path of NSynth dataset
        #Selecting model
        if args.model == "SC_CNN":  
            self.model = ShortChunkCNN_Res(n_channels=128, n_class=11).to(self.device)  
        elif args.model == "CNNSA":
            self.model = CNNSA(n_channels=128, n_class=11).to(self.device)  
        self.model.load_state_dict(torch.load(args.model_ckpt_path, weights_only=True))     #Load the best checkpoint params for the corresponding model
        self.model.eval()
        #The class and map for 11 kind of instruments
        self.classes = ["bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "synth_lead", "vocal"]
        self.label_map = {'bass': 0, 'brass': 1, 'flute': 2, 'guitar': 3, 'keyboard': 4, 'mallet': 5, 'organ': 6, 'reed': 7, 'string': 8, 'synth_lead': 9, 'vocal': 10}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}

    #Function for plotting confusion matrix
    def plot_conf_mtx(self, conf_mtx, classes, savepath):
        plt.figure(figsize=(10,7))
        sns.heatmap(conf_mtx, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=classes, yticklabels=classes)
        plt.title("Confusion matrix")
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(savepath)
    
    def criterion(self,predict,label):
        #Define loss function
        CE_loss = nn.CrossEntropyLoss()
        #Estimate total loss
        loss = CE_loss(predict,label)
        return loss.item()
    
    @torch.no_grad()
    def inference(self):
        json_path = os.path.join(self.testset_path, "examples.json")       #Metadata path of testset
        audio_path = os.path.join(self.testset_path, "audio")       #Path of .wav folder

        total_loss = 0.0
        total_top1_correct = 0.0
        total_top3_correct = 0.0
        count = 0
        audio_list = os.listdir(audio_path)     #The list of all wav file
        answer_list = []
        predict_list = []
        #Start testing
        with open(json_path,'r') as json_file:
            infos = json.load(json_file)
            for key, value in tqdm(infos.items(), desc="Testing", colour="#FF359A"):
                count += 1                      #For counting the number of data in testset
                #Getting the class name and wave path from the metadata
                class_name = value.get("instrument_family_str")
                wave_path = os.path.join(audio_path,key+".wav")

                #Load input & label
                wave, sr = librosa.load(wave_path, sr=None)
                label = [0] * len(self.label_map)  
                label[self.label_map[class_name]] = 1   #Create one-hot vector for estimating loss
                mel_spec = librosa.feature.melspectrogram(y=wave, sr=sr, win_length=1024, hop_length=256, n_mels=128)   #Transform to mel-spec
                #Transform to DB or not
                if args.use_log:
                    mel_spec = librosa.amplitude_to_db(mel_spec, ref=np.max)
                #Transform to tensor type
                mel_spec = torch.tensor(mel_spec)
                label = torch.tensor(np.array(label, dtype=np.float32))

                #Inference
                input = mel_spec.unsqueeze(0).to(self.device)
                label = label.unsqueeze(0).to(self.device)
                predict = self.model(input)     #Forward 
                
                #Est loss
                loss = self.criterion(predict,label)
                total_loss += loss
                #Est acc
                predict_class = torch.argmax(predict,dim=1)     #The top 1 predicted class
                predict_3class = torch.topk(predict, k=3, dim=1).indices    #The top 3 predicted class
                answer = torch.argmax(label, dim=1)     #Ground truth 
                #Inverse the class index back to instrument class name and save to list
                answer_list.append(self.inverse_label_map[answer.item()])      
                predict_list.append(self.inverse_label_map[predict_class.item()])   
                #Top1
                if predict_class == answer:
                    total_top1_correct += 1
                #Top3 
                if answer in predict_3class:
                    total_top3_correct += 1

        total_loss = total_loss/count
        total_top1_acc = total_top1_correct/count
        total_top3_acc = total_top3_correct/count
        conf_matrix = confusion_matrix(answer_list, predict_list)   #Estimating confusion matrix 
        print(f"avg loss: {total_loss} | avg top1 acc: {total_top1_acc} | avg top3 acc: {total_top3_acc}")  #Show acc and loss
        print("Confusion matrix: ")
        print(conf_matrix)
        self.plot_conf_mtx(conf_matrix,self.classes,args.mtx_savepath)      #Plot and save confusion matrix



#----------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset_path", type=str, default="/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-test", help="path of testdata")
    parser.add_argument("--model", type=str, default="SC_CNN", help="model name")
    parser.add_argument("--model_ckpt_path", type=str, default="./model_state/SC_CNN/use_log/best_model.pth", help="path of the best model ckpt")
    parser.add_argument("--mtx_savepath", type=str, default="./conf_mtx/", help="path to save confusion matrix")
    parser.add_argument("--use_log", action='store_true', help="Transform mel-spec to DB")
    args = parser.parse_args()

    #Set cuda
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
    #Build model
    model = Classifier(args, gpu_id=gpu_id)
    #Inference
    model.inference()