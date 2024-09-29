from model.dae import DAESkipConnections
from model.open_unmix import OpenUnmix

import os
import argparse
import torch
import torch.nn
import torchaudio
import numpy as np 
from mir_eval.separation import bss_eval_sources
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as sisdr
from tqdm import tqdm

def Inference(args):
    #Parameter setup
    resampler = torchaudio.transforms.Resample(44100, 16000)
    batch_size = 64
    n_fft = 2048
    win_length = 2048
    hop_length = 512
    frame_seg = 15
    stft = torchaudio.transforms.Spectrogram(n_fft=n_fft,win_length=win_length,hop_length=hop_length,power=None)
    istft = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    griffinlim = torchaudio.transforms.GriffinLim(n_fft=n_fft,win_length=win_length,hop_length=hop_length,power=1)
    #Model setup
    #model = DAESkipConnections().to('cuda')
    model = OpenUnmix(nb_bins = 1025,nb_channels = 1,hidden_size = 512,nb_layers = 6).to('cuda')
    #Load model params
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.eval()
    
    total_sdr = 0.0
    total_sisdr = 0.0
    track_count = 0
    track_list = os.listdir(args.test_path)
    for track in tqdm(track_list):
        track_count += 1
        # Audio path
        mixture_path = os.path.join(args.test_path,track,"mixture.wav")
        vocal_path = os.path.join(args.test_path,track,"vocals.wav")
        #Load audio
        mix_wav, sr = torchaudio.load(mixture_path)
        vocal_wav, sr = torchaudio.load(vocal_path)
        #Resample
        mix_wav, vocal_wav = resampler(mix_wav), resampler(vocal_wav)
        #Transform to mono channel
        mix_wav, vocal_wav = torch.mean(mix_wav,0,keepdim=True), torch.mean(vocal_wav,0,keepdim=True)        #G.T here
        #Time padding and STFT
        n_sample_points = mix_wav.shape[1]
        time_pad_size = (max(0, n_sample_points - win_length) // hop_length + 1) * hop_length - n_sample_points + win_length    #Crucial
        mix_wav_padded = torch.nn.functional.pad(mix_wav,(0, time_pad_size),mode='constant',value=0.0)
        mix_spec = stft(mix_wav_padded)
        mix_mag, mix_phase = torch.abs(mix_spec), torch.angle(mix_spec)             #Phase here
        #Freq padding and split
        segments = list(torch.split(mix_mag, frame_seg, dim=-1))
        freq_pad_size = 0
        if segments[-1].shape[-1] != frame_seg:
            freq_pad_size = frame_seg - segments[-1].shape[-1]
            segments[-1] = torch.nn.functional.pad(segments[-1], (0,freq_pad_size))
        mix_in = torch.stack(segments)          #Padded signal as input for sliding window
        #Start sliding window
        pred = []
        start_idx, end_idx = 0, batch_size    #64 refer to batch size
        while start_idx < mix_in.shape[0]:
            x = mix_in[start_idx:end_idx]
            with torch.no_grad():
                y = model(x.to('cuda'))
            pred.append(y.cpu())

            start_idx += batch_size
            end_idx += batch_size
            if end_idx > mix_in.shape[0]:
                end_idx = mix_in.shape[0]
        pred = torch.cat(pred, dim=0)
        pred_padded_spectrogram = torch.cat(torch.unbind(pred), dim=-1)
        pred_spec = pred_padded_spectrogram if freq_pad_size == 0 else pred_padded_spectrogram[:, :, 0: -freq_pad_size]
        #ISTFT back to waveform 
        if args.phase == "org":
            pred_padded_waveform = istft(torch.polar(pred_spec,mix_phase))
            final_pred_wave = pred_padded_waveform if time_pad_size == 0 else pred_padded_waveform[:, 0:-time_pad_size]
        elif args.phase == "griffin":
            pred_padded_waveform = griffinlim(pred_spec)
            final_pred_wave = pred_padded_waveform if time_pad_size == 0 else pred_padded_waveform[:, 0:-time_pad_size]
        
        #Save as output audio
        #torchaudio.save(os.path.join(args.out_path,track+".wav"),final_pred_wave,16000)
        
        #Est SDR
        ground_truth = vocal_wav.numpy()
        SDR, SIR, SAR, _ = bss_eval_sources(ground_truth, np.array(final_pred_wave))
        SISDR = sisdr()(vocal_wav, final_pred_wave)
        total_sdr += SDR
        total_sisdr += SISDR

    print(f"final sdr: {total_sdr/track_count} si-sdr: {total_sisdr/track_count}")



#------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="./model_state/DAE/best_128.ckpt", help='paths of models used for the inference for each channel')
    parser.add_argument('--test_path', type=str, default="/home/data1/dcn2001/MUSDBHQ/test")
    parser.add_argument('--out_path', type=str, default="./pred_out")
    parser.add_argument('--phase', type=str, default="org", help="org or griffin or vocoder")
    args = parser.parse_args()
    Inference(args)