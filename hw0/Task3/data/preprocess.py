import os
import json
import argparse
import numpy as np
import librosa
from tqdm import tqdm

#Preprocess for the trainset and validset to extract mel-spec
def preprocess(org_path,new_path):
    os.makedirs(new_path, exist_ok=True)                  #Create the new folder to save npy
    json_path = os.path.join(org_path,"examples.json")   #path of json
    audio_path = os.path.join(org_path,"audio")          #path of audio folder
    
    audio_list = os.listdir(audio_path)
    #Create synth_lead folder since validset and testset don't have this class of data
    if os.path.basename(new_path) == "test" or os.path.basename(new_path) == "valid":
        os.makedirs(os.path.join(new_path,"synth_lead"), exist_ok=True)
    #Use the info of json file to read data to get feature and move to the corresponding new folder
    with open(json_path,'r') as json_file:
        infos = json.load(json_file)
        for key, value in tqdm(infos.items(), desc="preprocessing", colour="#FF8000"):
            class_name = value.get("instrument_family_str")    #Get the instru class name
            os.makedirs(os.path.join(new_path,class_name), exist_ok=True)   #Create the class subfolder    

            wave_path = os.path.join(audio_path,key+".wav")    #Path of the audio data
            npy_path = os.path.join(new_path,class_name,key+".npy")  #Path to save the melspec (as .npy file)
            
            #Estimating mel-spec
            wave, sr = librosa.load(wave_path, sr=None)    #Load the audio to np.array by librosa
            mel_spec = librosa.feature.melspectrogram(y=wave, sr=sr, win_length=1024, hop_length=256, n_mels=128)    #Short hop length for temporal resolution
            np.save(npy_path, mel_spec)   #Save as npy
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--NSynth_path", type=str, default="/mnt/gestalt/home/dcn2001/NSynth_org", help="Original NSynth path")
    parser.add_argument("--new_path", type=str, default="/mnt/gestalt/home/dcn2001/NSynth_pre_task3", help="the new directory of processed data")
    args = parser.parse_args()

    os.makedirs(args.new_path,exist_ok=True)    #Create the nsynth feature folder
    nsynth_train_path, new_train_path = "/mnt/gestalt/home/dcn2001/nsynth-subtrain", os.path.join(args.new_path,"train")    #datapath for trainset
    nsynth_valid_path, new_valid_path = os.path.join(args.NSynth_path,"nsynth-valid"), os.path.join(args.new_path,"valid")  #datapath for validset
    
    #Start preprocessing
    preprocess(nsynth_train_path, new_train_path)
    preprocess(nsynth_valid_path, new_valid_path)