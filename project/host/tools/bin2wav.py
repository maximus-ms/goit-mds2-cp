#!/usr/bin/env python3
"""
Binary to WAV converter.
Binary file contains interleaved samples from multiple channels.
Supports 16-bit and 32-bit samples.
"""

import struct
import sys
import argparse
import os
import zlib
import ctypes


def calculate_crc32(data):
    """
    Calculates CRC32 checksum (as in ESP-IDF esp_rom_crc32_le).
    Uses zlib.crc32 with initial value 0xFFFFFFFF.
    """
    crc = zlib.crc32(data, 0xFFFFFFFF) & 0xFFFFFFFF
    return crc


def calculate_simple_sum(data):
    """
    Calculates simple sum of all bytes.
    """
    return sum(data)

def sample_processor(sample, sample_width):
    """
    Processes a sample.
    """
    b = sample
    if sample_width == 16:
        c = ctypes.c_int16(b).value
    else:
        c = ctypes.c_int32(b).value
    return c


def samples_normalizer(samples, sample_width_bits):
    """
    Normalizes samples to appropriate range depending on sample_width_bits.
    
    Args:
        samples: List of samples to normalize
        sample_width_bits: Sample width in bits (16 or 32)
    """
    max_sample = max(samples)
    min_sample = min(samples)
    max_abs = max(abs(max_sample), abs(min_sample))
    
    if sample_width_bits == 16:
        max_range = 32767
        min_range = -32768
    elif sample_width_bits == 32:
        max_range = 2147483647
        min_range = -2147483647
    else:
        max_range = 2147483647
        min_range = -2147483647
    
    if max_abs > 0:
        scale_factor = float(max_range) / max_abs
        normalized_samples = []
        cur_type = ctypes.c_int16 if sample_width_bits == 16 else ctypes.c_int32
        for sample in samples:
            normalized = int(sample * scale_factor)
            normalized = max(min_range, min(max_range, normalized))
            normalized_samples.append(cur_type(normalized).value)
        samples = normalized_samples

    return samples


def bin_to_wav(input_file, output_file, num_channels=2, sample_rate=44100, sample_width=32,
                expected_crc32=0):
    """
    Converts binary file to WAV format.
    
    Args:
        input_file: Path to input binary file
        output_file: Path to output WAV file
        num_channels: Number of channels (1=mono, 2=stereo, etc.)
        sample_rate: Sampling rate (Hz)
        sample_width: Sample width in bits (16 or 32)
        expected_crc32: Expected CRC32 checksum (hex string or int)
    """
    if not os.path.exists(input_file):
        print(f"Error: file '{input_file}' not found")
        return False
    
    with open(input_file, 'rb') as f:
        data = f.read()
    
    file_size = len(data)
    calculated_crc32 = calculate_crc32(data)
    calculated_sum = calculate_simple_sum(data)
    
    checksum_valid = True
    if expected_crc32 != 0:
        if isinstance(expected_crc32, str):
            expected_crc32 = expected_crc32.replace('0x', '').replace('0X', '')
            try:
                expected_crc32 = int(expected_crc32, 16)
            except ValueError:
                print(f"Error: invalid CRC32 format: {expected_crc32}")
                return False
        elif isinstance(expected_crc32, int):
            pass
        else:
            print(f"Error: invalid CRC32 type")
            return False
        
        if calculated_crc32 != expected_crc32:
            checksum_valid = False
            print("ERROR: Checksum does not match!")
            response = input("Continue conversion? (y/n): ")
            if response.lower() != 'y':
                return False
    
    sample_width_bytes = sample_width // 8
    sample_size = sample_width_bytes * num_channels
    if file_size % sample_size != 0:
        file_size = (file_size // sample_size) * sample_size
        data = data[:file_size]
    
    num_samples = file_size // sample_size
    
    samples = []
    unpack_format = {
        16: '<h',
        32: '<i'
    }
    
    if sample_width not in unpack_format:
        print(f"Error: unsupported sample width: {sample_width} bits")
        return False
    
    fmt = unpack_format[sample_width]
    for i in range(0, file_size, sample_width_bytes):
        sample = struct.unpack(fmt, data[i:i+sample_width_bytes])[0]
        samples.append(sample_processor(sample, sample_width))
    
    samples = samples_normalizer(samples, sample_width)
    
    with open(output_file, 'wb') as wav_file:
        wav_file.write(b'RIFF')
        wav_file.write(struct.pack('<I', 0))
        wav_file.write(b'WAVE')
        
        wav_file.write(b'fmt ')
        fmt_chunk_size = 16
        wav_file.write(struct.pack('<I', fmt_chunk_size))
        audio_format = 1
        wav_file.write(struct.pack('<H', audio_format))
        wav_file.write(struct.pack('<H', num_channels))
        wav_file.write(struct.pack('<I', sample_rate))
        byte_rate = sample_rate * num_channels * sample_width_bytes
        wav_file.write(struct.pack('<I', byte_rate))
        block_align = num_channels * sample_width_bytes
        wav_file.write(struct.pack('<H', block_align))
        bits_per_sample = sample_width
        wav_file.write(struct.pack('<H', bits_per_sample))
        
        wav_file.write(b'data')
        data_size = len(samples) * sample_width_bytes
        wav_file.write(struct.pack('<I', data_size))
        
        pack_format = {
            16: '<h',
            32: '<i'
        }
        
        fmt = pack_format[sample_width]
        for sample in samples:
            wav_file.write(struct.pack(fmt, sample))
        
        file_size_total = wav_file.tell() - 8
        wav_file.seek(4)
        wav_file.write(struct.pack('<I', file_size_total))
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Converts binary file with interleaved samples to WAV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  %(prog)s input.bin output.wav
  %(prog)s input.bin output.wav --channels 2 --rate 44100
  %(prog)s input.bin output.wav -c 1 -r 8000 -w 16
  %(prog)s input.bin output.wav --crc32 0x12345678
  %(prog)s input.bin output.wav -C 12345678 -w 16
        """
    )
    
    parser.add_argument('input', help='Input binary file')
    parser.add_argument('output', help='Output WAV file')
    parser.add_argument('-c', '--channels', type=int, default=2,
                        help='Number of channels (default: 2)')
    parser.add_argument('-r', '--rate', type=int, default=32000,
                        help='Sampling rate in Hz (default: 32000)')
    parser.add_argument('-w', '--width', type=int, default=32,
                        help='Sample width in bits (16 or 32, default: 32)')
    parser.add_argument('-C', '--crc32', type=str, default=0,
                        help='Expected CRC32 checksum (hex, e.g.: 0x12345678 or 12345678)')

    
    args = parser.parse_args()
    
    if args.channels < 1:
        print("Error: number of channels must be >= 1")
        sys.exit(1)
    
    if args.crc32 != 0:
        args.crc32 = eval(args.crc32)

    if args.rate < 1:
        print("Error: sampling rate must be >= 1")
        sys.exit(1)
    
    if args.width not in [16, 32]:
        print("Error: sample width must be 16 or 32 bits")
        sys.exit(1)
    
    success = bin_to_wav(
        args.input,
        args.output,
        num_channels=args.channels,
        sample_rate=args.rate,
        sample_width=args.width,
        expected_crc32=args.crc32,
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
