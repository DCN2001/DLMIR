import os
import json
import argparse
import numpy as np
import librosa
from scipy.signal import hilbert
from tqdm import tqdm

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
            npy_path = os.path.join(new_path,class_name,key+".npy")  #Path to save the extracted feature (as .npy file)
            
            #Extracting feature and get mean on the time axis
            wave, sr = librosa.load(wave_path, sr=None)    #Load the audio to np.array by librosa
            mfcc = librosa.feature.mfcc(y=wave, sr=sr).mean(axis=1)                     #MFCC feature
            spec_centroid = librosa.feature.spectral_centroid(y=wave,sr=sr).mean()      #spectrum centroid 
            spec_BW = librosa.feature.spectral_bandwidth(y=wave,sr=sr).mean()           #Bandwidth 
            spec_contrast = librosa.feature.spectral_contrast(y=wave,sr=sr).mean()      #Contrast
            spec_rolloff = librosa.feature.spectral_rolloff(y=wave,sr=sr).mean()        #Rolloff
            spec_flatness = librosa.feature.spectral_flatness(y=wave).mean()            #Flatness
            #Extracting ADSR as feature additionally
            envelope = np.abs(hilbert(wave))        #Get the envelop first
            max_amp = np.max(envelope)              #The maxima amplitude of wave
            attack_time_index = np.argmax(envelope)     #The time of max amp as attack time
            attack_time = attack_time_index / sr        
            sustain_level = 0.7 * max_amp               #Sustain Level
            decay_time_index = np.where(envelope <= sustain_level)[0][0]        #The time of decay  
            decay_time = (decay_time_index - attack_time_index) / sr
            #Estimate the sustain level but no rest cause it is hard to estimate
            sustain_level_time_start = decay_time_index            
            sustain_level_time_end = len(wave) - int(0.2 * sr)  
            sustain_level_value = np.mean(envelope[sustain_level_time_start:sustain_level_time_end])

            #Concatenate all the feature
            features = np.concatenate([ mfcc, np.array([spec_centroid]), np.array([spec_BW]), 
                                       np.array([spec_contrast]), np.array([spec_rolloff]), np.array([spec_flatness]),
                                       np.array([attack_time]), np.array([decay_time]), np.array([sustain_level_value])])

            np.save(npy_path, features)   #Save as npy
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--NSynth_path", type=str, default="/mnt/gestalt/home/dcn2001/NSynth_org", help="Original NSynth path")
    parser.add_argument("--new_path", type=str, default="/mnt/gestalt/home/dcn2001/NSynth_pre_task2", help="the new directory of processed data")
    args = parser.parse_args()

    os.makedirs(args.new_path,exist_ok=True)    #Create the nsynth feature folder

    #Setting all the path of trainset, validset and testset
    nsynth_train_path, new_train_path = "/mnt/gestalt/home/dcn2001/nsynth-subtrain", os.path.join(args.new_path,"train")
    nsynth_valid_path, new_valid_path = os.path.join(args.NSynth_path,"nsynth-valid"), os.path.join(args.new_path,"valid")
    nsynth_test_path, new_test_path = os.path.join(args.NSynth_path,"nsynth-test"), os.path.join(args.new_path,"test")
    
    #Start preprocessing...
    preprocess(nsynth_train_path, new_train_path)
    preprocess(nsynth_valid_path, new_valid_path)
    preprocess(nsynth_test_path, new_test_path)