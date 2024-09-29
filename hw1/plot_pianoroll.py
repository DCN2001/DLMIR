import matplotlib.pyplot as plt
from glob import glob
import os
import json
import pretty_midi
import numpy as np
import argparse


with open('./map/class_idx2MIDIClass.json') as f:
    class_idx2MIDIClass = json.load(f)
with open('./map/idx2instrument_class.json') as f:
    idx2instrument_class = json.load(f)
with open('./map/MIDIClassName2class_idx.json') as f:
    MIDIClassName2class_idx = json.load(f)

categories = [
    'Piano', 'Percussion', 'Organ', 'Guitar', 'Bass', 
    'Strings', 'Voice', 'Wind Instruments', 'Synth'
    ]

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_track_dir', type=str,
                        help='source(test) midi track folder path', default='./test_track')
    parser.add_argument('--save_dir', type=str,
                        help='saved fig folder path', default='./')
    args = parser.parse_args()
    return args

def extract_pianoroll_from_midi(midi_file_path, time_step=5.0):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    # print(midi_data)
    
    # Determine total duration in seconds
    total_time = midi_data.get_end_time()
    
    # Create an empty pianoroll matrix without the "Empty" class
    num_classes = len(class_idx2MIDIClass)
    num_time_steps = int(np.ceil(total_time / time_step))
    pianoroll = np.zeros((num_classes, num_time_steps))
    
    # Process each instrument in the MIDI file
    for instrument in midi_data.instruments:
        program_num = instrument.program
        
        if instrument.is_drum:
            instrument_class = 128
        else:
            # Determine the class for this instrument
            instrument_class = idx2instrument_class.get(str(program_num), None)
        if instrument_class and instrument_class in MIDIClassName2class_idx:
            class_idx = MIDIClassName2class_idx[instrument_class]
            
            # Fill the pianoroll for each note
            for note in instrument.notes:
                start_time = note.start
                end_time = note.end
                start_idx = int(np.floor(start_time / time_step))
                end_idx = int(np.ceil(end_time / time_step))
                pianoroll[class_idx, start_idx:end_idx] = 1  # Mark the note as present
    
    return pianoroll

def pianoroll_comparison(true_pianoroll, pred_pianoroll, save_path):
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    # Plotting the true pianoroll
    axes[0].imshow(true_pianoroll, aspect='auto', cmap='Oranges', interpolation='nearest')
    axes[0].set_title('True Labels')
    axes[0].set_yticks(range(len(categories)))
    axes[0].set_yticklabels(categories)
    axes[0].set_xlabel('Time Steps')

    # Plotting the predicted pianoroll
    axes[1].imshow(pred_pianoroll, aspect='auto', cmap='Oranges', interpolation='nearest')
    axes[1].set_title('Predicted Labels')
    axes[1].set_yticks(range(len(categories)))
    axes[1].set_yticklabels(categories)
    axes[1].set_xlabel('Time Steps')

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)

def main(opt):
    midi_path_list = glob(os.path.join(opt.test_track_dir, '*.mid'))
    audio_path_list = glob(os.path.join(opt.test_track_dir, '*.flac'))
    
    for src_path in midi_path_list:
        name = src_path.split('/')[-1].split('.')[0]
        true_pianoroll = extract_pianoroll_from_midi(src_path)
        # pred_pianoroll is your model predict result please load your results here
        # pred_pianoroll.shape should be [9, L] and the L should be equal to true_pianoroll
        pred_pianoroll = np.zeros(true_pianoroll.shape)
        pianoroll_comparison(true_pianoroll, pred_pianoroll, name+'.png')
    pass

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)