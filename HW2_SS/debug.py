import os
import random
import mir_eval
import torch
import torchaudio
import torch.utils.data
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np
from tqdm import tqdm


class Validset(torch.utils.data.Dataset):
    def __init__(self, validset_dir):
        self.datapath = validset_dir
        self.mixture_dir = os.path.join(validset_dir,"mixture")
        self.vocal_dir = os.path.join(validset_dir,"vocal")

        self.mixture_list = os.listdir(self.mixture_dir)
        self.vocal_list = os.listdir(self.vocal_dir)

        #self.to_DB = 20.0 * torch.log10(torch.clamp(amplitude, min=1e-10))
    
    def __len__(self):
        return len(self.mixture_list)

    def __getitem__(self, idx):
        mixture_path, vocal_path = os.path.join(self.mixture_dir,self.mixture_list[idx]), os.path.join(self.vocal_dir,self.vocal_list[idx])
        mixture_wave, vocal_wave = torch.tensor(np.load(mixture_path)), torch.tensor(np.load(vocal_path))
        mixture_wave, vocal_wave = mixture_wave[:,:22050*6], vocal_wave[:,:22050*6]
        #Transform input to STFT-spec
        # mix_spec_left = torch.stft(mixture_wave[0], n_fft=2048, window=torch.hamming_window(1024), win_length=1024, hop_length=512, return_complex=True)  
        # mix_spec_right = torch.stft(mixture_wave[1], n_fft=2048, window=torch.hamming_window(1024), win_length=1024, hop_length=512, return_complex=True)   
        # mixture_spec = torch.stack([torch.abs(mix_spec_left), torch.abs(mix_spec_right)], dim=0)
        # mixture_phase = torch.stack([torch.angle(mix_spec_left), torch.angle(mix_spec_right)], dim=0)

        #Transform target to STFT-spec
        n_fft = 4096
        n_hop = 1024
        gt_left = torch.stft(vocal_wave[0], n_fft=n_fft, window=torch.hamming_window(n_fft), hop_length=n_hop, return_complex=True, center=True, normalized=False, onesided=True, pad_mode="reflect")
        gt_right = torch.stft(vocal_wave[1], n_fft=n_fft, window=torch.hamming_window(n_fft), hop_length=n_hop, return_complex=True, center=True, normalized=False, onesided=True, pad_mode="reflect")
        vocal_spec = torch.stack([torch.abs(gt_left), torch.abs(gt_right)], dim=0)
        vocal_phase = torch.stack([torch.angle(gt_left), torch.angle(gt_right)], dim=0)

        #mixture_spec = self.to_DB(mixture_spec)
        vocal_spec = 20.0 * torch.log10(torch.clamp(vocal_spec, min=1e-10))
        return vocal_spec, vocal_phase, vocal_wave





def est_metric(vocal_out, vocal_label):
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
                batch_sdr += sdr 
            else:
                pass
        return batch_sdr/count

datapath = "/home/data1/dcn2001/MUSDBHQ_2/valid_chunk"
valid_ds = Validset(datapath)
validset_loader = torch.utils.data.DataLoader(dataset=valid_ds,
                                                batch_size=1,
                                                pin_memory=True,
                                                shuffle=False,
                                                drop_last=True,
                                                num_workers=8)

sdr_total = 0.0
for idx, batch in enumerate(tqdm(validset_loader)):
    step = idx + 1
    vocal_spec = batch[0]
    vocal_phase = batch[1]
    vocal_wave = batch[2]

    vocal_spec = F.DB_to_amplitude(vocal_spec,1.0,0.5)
    vocal_gt_left_complex = vocal_spec[:,0,:,:] * torch.exp(1j * vocal_phase[:,0,:,:])
    vocal_gt_right_complex = vocal_spec[:,1,:,:] * torch.exp(1j * vocal_phase[:,1,:,:])
    n_fft = 4096
    n_hop = 1024
    vocal_gt_wave_left = torch.istft(vocal_gt_left_complex, n_fft=n_fft, window=torch.hamming_window(n_fft), hop_length=n_hop, center=True, normalized=False, onesided=True, length=22050*10)
    vocal_gt_wave_right = torch.istft(vocal_gt_right_complex, n_fft=n_fft, window=torch.hamming_window(n_fft), hop_length=n_hop, center=True, normalized=False, onesided=True, length=22050*10)
    vocal_gt_wave = torch.stack([vocal_gt_wave_left, vocal_gt_wave_right], dim=1)
    #print(vocal_wave.shape, vocal_gt_wave.shape)
    #vocal_wave = vocal_wave[:,:,:vocal_gt_wave.shape[2]]

    torchaudio.save("./listen_data/vocal_trans/"+str(idx)+".wav", vocal_gt_wave.squeeze(0), 22050)
    torchaudio.save("./listen_data/vocal/"+str(idx)+".wav", vocal_wave.squeeze(0), 22050)
    #sdr_batch = est_metric(vocal_gt_wave, vocal_wave)
    #sdr_total += sdr_batch

print(f"sdr: {sdr_total/step}")




