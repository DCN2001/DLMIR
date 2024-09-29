from model.MERT_Classifier import MERT_Classifier
from plot_pianoroll import *

import os
import torch
import torchaudio
import numpy as np
import argparse
from tqdm import tqdm


@torch.no_grad()
def infererence_a_track(classifier,track_path,args):
    #Midi Path
    track_name = os.path.basename(track_path).replace('.flac','')   #Track name 
    midi_path = track_path.replace('.flac', '.mid')

    #Load Audio
    wave, sr = torchaudio.load(track_path)
    
    #Estimate the number of segments
    track_length = wave.shape[1]
    seg_length = sr * 5
    
    num_seg = track_length // seg_length    ##Since I found that the ground truth provided by TA have abandomed the last seg < 5 sec  
    predict_list = []
    #Sliding window
    for idx in tqdm(range(num_seg), colour="red"):
        #Select the window
        start_pt, end_pt = idx*seg_length, (idx+1)*seg_length
        wave_seg = wave[:,start_pt:end_pt].to('cuda')

        #Feed to model
        output = classifier(wave_seg)
        probs = torch.sigmoid(output).cpu().numpy()    #Logits to prob
        predict = (probs >= args.threshold).astype(int)     #prob to 0 or 1
        predict_list.append(predict)

    #Plot piano roll
    true_pianoroll = extract_pianoroll_from_midi(midi_path)         #Function provided by TA
    pred_pianoroll = np.array(predict_list).squeeze().reshape(num_seg, 9).T      #Squeeze the dimension and reshape to the shape of true_pianoroll 
    pianoroll_comparison(true_pianoroll, pred_pianoroll, os.path.join(args.pianoroll_path,track_name+'.png'))  #Function provided by TA
        

#--------------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track_folder", type=str, default="/mnt/gestalt/home/dcn2001/hw1/test_track")
    parser.add_argument("--model_path", type=str, default="./model_state/best_model_finetune.pth")
    parser.add_argument("--pianoroll_path", type=str, default="./result/pianoroll/finetune", help="Path to save pianoroll")
    parser.add_argument("--threshold", type=float, default="0.4", help="Threshold of transform prob to [0,1]")
    args = parser.parse_args()

    #Build & load model
    classifier = MERT_Classifier(n_class=9,freeze=False).to('cuda')
    classifier.load_state_dict(torch.load(args.model_path, weights_only=True))
    classifier.eval()
    
    #Inference on the test tracks (5 songs)
    tracks = [f for f in os.listdir(args.track_folder) if f.endswith('.flac')]
    for track in tracks:
        infererence_a_track(classifier,os.path.join(args.track_folder,track),args)
