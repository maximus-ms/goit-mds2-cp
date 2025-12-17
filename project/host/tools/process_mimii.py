#!/usr/bin/env python3
"""
Preprocess MIMII dataset: convert WAV files to PyTorch tensor format.
Processes multi-channel audio files, mixes selected channels, and saves as .pt file.
"""

import torch
import glob
import time
import os
import argparse
from tqdm import tqdm
import soundfile as sf


OUTPUT_FILE = "mimii.pt"
TARGET_SR = 16000

DEV2CHANNELS = {
    'fan': (3, 5),
    'pump': (1, 3),
    'slider': (5, 7),
    'valve': (1, 7)
}

SNR = ['6db', '0db', '_6db']
SNR6DB = '6db'


def get_channels_for_machine_type(machine_type):
    return DEV2CHANNELS[machine_type]


def preprocess_dataset(path_in, path_out=None):
    start_time = time.time()
    nested_data = {}
    
    if path_out is None:
        path_out = os.path.join(path_in, OUTPUT_FILE)
        
    dev_type_paths = glob.glob(os.path.join(path_in, SNR6DB, '*'))
    machine_types = [os.path.basename(p) for p in dev_type_paths if os.path.isdir(p)]
    
    print(f"Found machine types: {machine_types}")

    for machine_type in machine_types:
        print(f"Processing type: {machine_type}")
        nested_data[machine_type] = {}
        
        machine_id_paths = glob.glob(os.path.join(path_in, SNR6DB, machine_type, 'id_*'))
        machine_ids = [os.path.basename(p) for p in machine_id_paths if os.path.isdir(p)]
        target_channels = get_channels_for_machine_type(machine_type)

        for machine_id in machine_ids:
            nested_data[machine_type][machine_id] = {snr: [] for snr in SNR}

            for snr in SNR:
                wav_files = glob.glob(os.path.join(path_in, snr, machine_type, machine_id, 'normal', '*.wav'))
                
                for f_path in tqdm(wav_files, desc=f"Processing {machine_type} {machine_id} {snr}"):
                    try:
                        data, sr = sf.read(f_path)
                        
                        if sr != TARGET_SR:
                            raise ValueError(f"Invalid sample rate: {sr} != {TARGET_SR}")
                        
                        waveform = torch.from_numpy(data).float().t()
                        
                        ch_a = waveform[target_channels[0]]
                        ch_b = waveform[target_channels[1]]
                        mixed = (ch_a + ch_b) / 2.0
                        
                        mixed_int16 = (mixed * 32767).to(torch.int16)
                        nested_data[machine_type][machine_id][snr].append(mixed_int16.unsqueeze(0))
                        
                    except Exception as e:
                        print(f"Error processing {f_path}: {e}")

    print(f"Saving to {path_out}...")
    torch.save(nested_data, path_out)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess MIMII dataset: convert WAV files to PyTorch tensor format'
    )
    parser.add_argument('input', help='Input directory path containing MIMII dataset')
    parser.add_argument('-o', '--output', default=None,
                        help='Output file path (default: input/mimii.pt)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return 1
    
    preprocess_dataset(args.input, args.output)
    return 0


if __name__ == '__main__':
    exit(main())
