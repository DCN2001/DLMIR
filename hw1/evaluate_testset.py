from data.slakh_loader import Slakh
from model.MERT_Classifier import MERT_Classifier

import os
import torch
import numpy as np
import argparse
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm


@torch.no_grad()
def eval(args):
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
    classifier.load_state_dict(torch.load(args.model_path, weights_only=True))
    classifier.eval()

    predict_list = []
    label_list = []
    #Start testing
    for idx, batch in enumerate(tqdm(test_loader, desc="Testing", colour="red")):
        #Load input and label
        input = batch[0].to('cuda')
        label = batch[1]

        #Feed input to model
        outputs = classifier(input)
        probs = torch.sigmoid(outputs).cpu().numpy()

        #Convert to [0,1]
        predict = (probs >= args.threshold).astype(int)

        #Append to list
        predict_list.append(predict)
        label_list.append(label)
    
    #Flatten
    predicts = np.concatenate(predict_list,axis=0)
    labels = np.concatenate(label_list, axis=0)

    #Est classification report
    report = classification_report(labels, predicts, zero_division=0)
    print(report)

    #Save classification report as image
    plt.figure(figsize=(3, 4))
    plt.text(0.01, 1.0, str('Classification Report'), {'fontsize': 10,'weight': 'bold'}, fontproperties='monospace') 
    plt.text(0.01, 0.05, str(report), {'fontsize': 10}, fontproperties='monospace') 
    plt.axis('off')
    plt.savefig(args.report_path, dpi=300, bbox_inches='tight',facecolor='lightblue')
    plt.show()
    
        
#-----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/gestalt/home/dcn2001/hw1/slakh")
    parser.add_argument("--model_path", type=str, default="./model_state/best_model_finetune.pth", help="Path of model weight") 
    parser.add_argument("--report_path", type=str, default="./result/classification_report/best_model_finetune.png")  #Path to save image
    parser.add_argument("--threshold", type=float, default="0.4", help="Threshold of transform prob to [0,1]")
    args = parser.parse_args()

    eval(args)
    
    