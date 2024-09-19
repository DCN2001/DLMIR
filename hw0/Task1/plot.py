import matplotlib.pyplot as plt
import librosa
import numpy as np


def Mel_Spec(datapath, save_path):
    wave, sr = librosa.load(datapath, sr=None)
    spec = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=2048, hop_length=512)     #Keep other mel-spectrogram params as default
    spec_DB = librosa.power_to_db(spec, ref=np.max)

    #Plot the spectrogram 
    plt.figure()
    librosa.display.specshow(spec_DB, sr=sr, x_axis='time', y_axis='mel', hop_length=512)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')
    plt.savefig(save_path, bbox_inches='tight')


#------------------------------------------
# Execute Region
if __name__ == '__main__':
    #Choose the wave file & saved path
    paths_list = [["/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-train/audio/guitar_acoustic_001-025-050.wav","./result/guitar_pitch25.jpg"],\
                  ["/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-train/audio/guitar_acoustic_001-050-100.wav", "./result/guitar_pitch50.jpg"],\
                  ["/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-train/audio/guitar_acoustic_001-060-050.wav", "./result/guitar_pitch60.jpg"],\
                  ["/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-train/audio/bass_acoustic_000-025-050.wav", "./result/bass_pitch25.jpg"],\
                  ["/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-train/audio/bass_acoustic_000-050-100.wav", "./result/bass_pitch50.jpg"],\
                  ["/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-train/audio/bass_acoustic_000-060-050.wav", "./result/bass_pitch60.jpg"],\
                  ["/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-train/audio/vocal_acoustic_011-038-025.wav", "./result/vocal_pitch38.jpg"],\
                  ["/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-train/audio/vocal_acoustic_011-050-100.wav", "./result/vocal_pitch50.jpg"],\
                  ["/mnt/gestalt/home/dcn2001/NSynth_org/nsynth-train/audio/vocal_acoustic_011-060-050.wav", "./result/vocal_pitch60.jpg"]]
    
    for path in paths_list:
        Mel_Spec(path[0],path[1])
    
    

    