import os 
import numpy as np
from tqdm import tqdm

datapath = "/home/data1/dcn2001/MUSDBHQ_2/test"
newpath = "/home/data1/dcn2001/MUSDBHQ_va/valid_chunk"

valid_song = [  'Actions - One Minute Smile','Clara Berry And Wooldog - Waltz For My Victims',
                'Johnny Lokke - Promises & Lies','Patrick Talbot - A Reason To Leave',
                'Triviul - Angelsaint','Alexander Ross - Goodbye Bolero',
                'Fergessen - Nos Palpitants','Leaf - Summerghost',
                'Skelpolu - Human Mistakes','Young Griffo - Pennies',
                'ANiMAL - Rockshow','James May - On The Line',
                'Meaxic - Take A Step','Traffic Experiment - Sirens']


#Valid
track_list = os.listdir(datapath)
chunk_length = 22050 * 10
for track in track_list:
    if track in valid_song:
        mixture_wave = np.load(os.path.join(datapath,track,"mixture.npy"))
        vocal_wave = np.load(os.path.join(datapath,track,"vocals.npy"))
        track_length = mixture_wave.shape[1]    
        num_chunk = track_length // chunk_length
        for idx in range(num_chunk):
            start_pt = chunk_length * idx
            end_pt = start_pt + chunk_length
            mixture_chunk = mixture_wave[:,start_pt:end_pt]
            vocal_chunk = vocal_wave[:,start_pt:end_pt]
            np.save(os.path.join(newpath,"mixture",track+"_"+str(idx)+".npy"),mixture_chunk)
            np.save(os.path.join(newpath,"vocal",track+"_"+str(idx)+".npy"),vocal_chunk)


'''
track_list = os.listdir(datapath)
chunk_length = 22050 * 10
for track in tqdm(track_list):
    mixture_wave = np.load(os.path.join(datapath,track,"mixture.npy"))
    vocal_wave = np.load(os.path.join(datapath,track,"vocals.npy"))
    track_length = mixture_wave.shape[1]    
    num_chunk = track_length // chunk_length
    for idx in range(num_chunk):
        start_pt = chunk_length * idx
        end_pt = start_pt + chunk_length
        mixture_chunk = mixture_wave[:,start_pt:end_pt]
        vocal_chunk = vocal_wave[:,start_pt:end_pt]
        np.save(os.path.join(newpath,"mixture",track+"_"+str(idx)+".npy"),mixture_chunk)
        np.save(os.path.join(newpath,"vocal",track+"_"+str(idx)+".npy"),vocal_chunk)

'''

