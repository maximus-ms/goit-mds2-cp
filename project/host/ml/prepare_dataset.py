#!/usr/bin/env python3
"""
Audio Dataset Preprocessor

This script processes audio datasets with MIMII-like structure and prepares them for training.
It supports both MIMII dataset and custom datasets with the same directory layout:
    ROOT_DIR / snr / machine_type / machine_id / normal|abnormal / *.wav

It can be run as a standalone script or imported as a module.

Configuration:
    The script supports configuration via environment variables or .env file.
    See env.example for available configuration options.
    If python-dotenv is installed, .env file will be automatically loaded.

Usage as script:
    python prepare_dataset.py <input_path> [--output <output_path>] [--abnormal] [--jobs <n_jobs>]

Usage as module:
    from prepare_dataset import preprocess_dataset, load_dataset
    dataset = preprocess_dataset('path/to/data', 'output.pt', use_abnormal=True)
    # or load existing dataset
    dataset = load_dataset('output.pt')
"""

import torch
import glob
import time
import os
import argparse
import sys
import logging
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from collections import defaultdict
import soundfile as sf

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants (can be overridden via environment variables)
DEFAULT_DATASET_FILE = os.getenv('DEFAULT_DATASET_FILE', 'dataset.pt')
TARGET_SR = int(os.getenv('DATASET_TARGET_SR', '16000'))

# MIMII dataset channel mapping (optional, can be overridden)
# Format: {machine_type: (channel_a, channel_b)}
DEFAULT_DEV2CHANNELS = {
    'fan': (3, 5),
    'pump': (1, 3),
    'slider': (5, 7),
    'valve': (1, 7)
}

# Default SNR values for MIMII dataset (optional, auto-detected if not provided)
DEFAULT_SNR_VALUES = ['6db', '0db', '_6db']


def get_channels_for_machine_type(machine_type: str, channel_mapping: Optional[Dict[str, Tuple[int, int]]] = None) -> Tuple[int, int]:
    """
    Get target channels for a given machine type.
    
    Args:
        machine_type: Type of machine (fan, pump, slider, valve, or custom)
        channel_mapping: Optional custom channel mapping dictionary.
                        If None, uses DEFAULT_DEV2CHANNELS or auto-detects channels.
    
    Returns:
        Tuple of two channel indices to use for mixing.
        Defaults to (0, 1) if machine type not found in mapping.
    """
    if channel_mapping is None:
        channel_mapping = DEFAULT_DEV2CHANNELS
    
    if machine_type in channel_mapping:
        return channel_mapping[machine_type]
    
    # For unknown machine types, default to first two channels
    logger.warning(f"Unknown machine type '{machine_type}', using default channels (0, 1)")
    return (0, 1)


def load_and_process_wav_file(wav_path: str, sample_rate: int = TARGET_SR,
                              target_channels: Tuple[int, int] = (0, 1)) -> torch.Tensor:
    """
    Load and process a WAV file from disk.
    This function is used for dataset preprocessing and ensures consistent audio processing
    across all parts of the pipeline (preprocessing, training, testing).
    
    Args:
        wav_path: Path to WAV file
        sample_rate: Target sample rate (default: TARGET_SR = 16000)
        target_channels: Tuple of two channel indices to mix (default: (0, 1))
    
    Returns:
        Processed waveform tensor in int16 format [1, samples]
    
    Raises:
        ValueError: If sample rate doesn't match or insufficient channels
        ImportError: If soundfile is not installed
    """
    # Load audio file using soundfile
    data, sr = sf.read(wav_path)
    
    # soundfile returns shape [Time, Channels] (e.g., 160000, 8)
    # PyTorch expects [Channels, Time] (8, 160000)
    # Therefore we need to transpose the matrix (.t())
    waveform = torch.from_numpy(data).float().t()
    
    # Check sample rate
    if sr != sample_rate:
        raise ValueError(f"Incorrect sample rate: {sr} != {sample_rate}")
    
    # Check if we have enough channels
    num_channels = waveform.shape[0]
    if max(target_channels) >= num_channels:
        raise ValueError(f"Not enough channels: file has {num_channels}, "
                       f"but requested channels {target_channels}")
    
    # Mix channels
    ch_a = waveform[target_channels[0]]
    ch_b = waveform[target_channels[1]]
    mixed = (ch_a + ch_b) / 2.0
    
    # Convert float32 -> int16 (memory efficiency)
    # Multiply by 32767 to preserve amplitude when casting to int
    mixed_int16 = (mixed * 32767).to(torch.int16)
    
    # Add channel dimension: [samples] -> [1, samples]
    return mixed_int16.unsqueeze(0)


def process_single_file(f_path: str, machine_type: str, machine_id: str, snr: str, 
                       normal: str, target_channels: Tuple[int, int]) -> Optional[Dict]:
    """
    Process a single audio file and return processed data.
    This function runs in a separate process for parallel execution.
    
    Uses load_and_process_wav_file to ensure consistency with dataset loading logic.
    
    Args:
        f_path: Path to the WAV file
        machine_type: Type of machine (fan, pump, slider, valve, or custom)
        machine_id: Machine ID (e.g., id_00, id_02)
        snr: SNR level (e.g., 6db, 0db, _6db)
        normal: 'normal' or 'abnormal'
        target_channels: Tuple of two channel indices to mix
    
    Returns:
        Dictionary with processed data and metadata, or None if processing failed
    """
    try:
        # Use load_and_process_wav_file for loading and processing WAV file
        # This ensures exact same logic as used during training/testing
        mixed_int16 = load_and_process_wav_file(
            f_path,
            sample_rate=TARGET_SR,
            target_channels=target_channels
        )
        
        # Extract numeric key from filename (e.g., "00000001.wav" -> 1)
        filename = os.path.basename(f_path)
        filename_without_ext = os.path.splitext(filename)[0]
        try:
            file_key = int(filename_without_ext)
        except ValueError:
            # If filename doesn't contain a number, use hash as fallback
            file_key = hash(filename_without_ext) % 1000000
            logger.warning(f"Could not extract numeric key from filename '{filename}', using hash: {file_key}")
        
        # Return result with metadata for collection
        return {
            'machine_type': machine_type,
            'machine_id': machine_id,
            'snr': snr,
            'normal': normal,
            'data': mixed_int16,  # Already has shape [1, samples]
            'file_key': file_key,
            'file_path': f_path
        }
    except Exception as e:
        logger.error(f"Error processing {f_path}: {e}")
        return None


def preprocess_dataset(path_in: str, path_out: Optional[str] = None, 
                     n_jobs: Optional[int] = None, use_abnormal: bool = False,
                     channel_mapping: Optional[Dict[str, Tuple[int, int]]] = None,
                     snr_values: Optional[List[str]] = None,
                     auto_detect_snr: bool = True) -> Dict:
    """
    Parallel version of dataset preprocessing.
    
    Processes all WAV files in a MIMII-like dataset structure and saves them
    as a single PyTorch file. The output structure is:
    {
        machine_type: {
            machine_id: {
                snr: {
                    'normal': {file_key: tensor, ...},  # file_key is numeric from filename
                    'abnormal': {file_key: tensor, ...}  # if use_abnormal=True
                }
            }
        }
    }
    
    Args:
        path_in: Path to root directory containing the dataset
        path_out: Path to output file (None = path_in/dataset.pt)
        n_jobs: Number of parallel processes (None = auto = number of CPU cores)
        use_abnormal: Whether to include abnormal samples (default: False)
        channel_mapping: Optional custom channel mapping dictionary.
                         If None, uses DEFAULT_DEV2CHANNELS.
        snr_values: Optional list of SNR values to validate against.
                   If None and auto_detect_snr=False, uses DEFAULT_SNR_VALUES.
        auto_detect_snr: If True, automatically detects SNR directories (default: True)
    
    Returns:
        Dictionary containing the processed dataset
    """
    start_time = time.time()
    
    if n_jobs is None:
        n_jobs = cpu_count() - 2
        if n_jobs < 1: n_jobs = 1
    
    if path_out is None:
        path_out = os.path.join(path_in, DEFAULT_DATASET_FILE)
    
    logger.info(f"Using {n_jobs} processes for parallel processing")
    logger.info(f"Input path: {path_in}")
    logger.info(f"Output path: {path_out}")
    
    # Main nested data structure
    # Changed from list to dict: samples are stored as dict with numeric keys from filenames
    nested_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    
    # 1. Collect all files for processing
    snrs = glob.glob(os.path.join(path_in, '*db'))
    snrs = [os.path.basename(p) for p in snrs if os.path.isdir(p)]
    
    if not snrs:
        # Try alternative patterns
        snrs = glob.glob(os.path.join(path_in, '*'))
        snrs = [os.path.basename(p) for p in snrs if os.path.isdir(p) and not p.endswith('.pt')]
        logger.warning(f"No '*db' directories found, using all directories: {snrs}")
    
    # Validate SNR values if provided
    if snr_values is not None and not auto_detect_snr:
        invalid_snrs = [snr for snr in snrs if snr not in snr_values]
        if invalid_snrs:
            logger.warning(f"Found SNR directories not in provided list: {invalid_snrs}")
    elif snr_values is None and not auto_detect_snr:
        snr_values = DEFAULT_SNR_VALUES
        logger.info(f"Using default SNR values: {snr_values}")
    
    logger.info(f"Found SNR levels: {snrs}")
    
    # Structure: ROOT_DIR / snr / machine_type / machine_id / normal|abnormal / *.wav
    if not snrs:
        raise ValueError(f"No directories found in {path_in}")
    
    dev_type_paths = glob.glob(os.path.join(path_in, snrs[0], '*'))
    machine_types = [os.path.basename(p) for p in dev_type_paths if os.path.isdir(p)]
    
    if not machine_types:
        raise ValueError(f"No machine type directories found in {os.path.join(path_in, snrs[0])}")
    
    logger.info(f"Found machine types: {machine_types}")
    
    # Create list of tasks (separate arguments for process_single_file)
    tasks = []
    
    for machine_type in machine_types:
        logger.info(f"Preparing tasks for type: {machine_type}")
        
        machine_id_paths = glob.glob(os.path.join(path_in, snrs[0], machine_type, 'id_*'))
        machine_ids = [os.path.basename(p) for p in machine_id_paths if os.path.isdir(p)]
        
        # If no 'id_*' pattern found, try to find any subdirectories
        if not machine_ids:
            all_paths = glob.glob(os.path.join(path_in, snrs[0], machine_type, '*'))
            machine_ids = [os.path.basename(p) for p in all_paths if os.path.isdir(p)]
            logger.info(f"No 'id_*' pattern found for {machine_type}, using all subdirectories: {machine_ids}")
        
        try:
            target_channels = get_channels_for_machine_type(machine_type, channel_mapping)
        except Exception as e:
            logger.warning(f"Error getting channels for {machine_type}: {e}, using default (0, 1)")
            target_channels = (0, 1)
        
        logger.info(f"    Found IDs: {machine_ids}, target_channels: {target_channels}")
        
        for machine_id in machine_ids:
            for snr in snrs:
                # Get files from 'normal' and optionally 'abnormal' folders
                normal_abnormal = ['normal', 'abnormal'] if use_abnormal else ['normal']
                for normal in normal_abnormal:
                    wav_files = glob.glob(os.path.join(path_in, snr, machine_type, machine_id, normal, '*.wav'))
                    wav_files.sort()
                    
                    # Add each file as a separate task
                    for f_path in wav_files:
                        tasks.append((f_path, machine_type, machine_id, snr, normal, target_channels))
    
    logger.info(f"Total files to process: {len(tasks)}")
    
    if len(tasks) == 0:
        raise ValueError(f"No WAV files found in {path_in}")
    
    # 2. PARALLEL PROCESSING of files
    # Use joblib.Parallel for parallel processing (works better with Jupyter)
    logger.info("Processing files...")
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_file)(f_path, machine_type, machine_id, snr, normal, target_channels)
        for f_path, machine_type, machine_id, snr, normal, target_channels in tqdm(tasks, desc="Preparing tasks")
    )
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    if len(results) == 0:
        raise ValueError("No files were successfully processed!")
    
    logger.info(f"Successfully processed {len(results)} out of {len(tasks)} files")
    
    # 3. COLLECT RESULTS into dictionary (sequential, but this is fast)
    logger.info("Collecting results...")
    for result in results:
        # Store data in dict with numeric key from filename instead of list
        nested_data[result['machine_type']][result['machine_id']][result['snr']][result['normal']][result['file_key']] = result['data']
    
    # Convert defaultdict to regular dict for saving
    nested_data = {
        mt: {
            mid: {
                snr: dict(normal_abnormal_data)
                for snr, normal_abnormal_data in snr_data.items()
            }
            for mid, snr_data in mid_data.items()
        }
        for mt, mid_data in nested_data.items()
    }
    logger.info(f"Saving to file {path_out}...")
    torch.save(nested_data, path_out)
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info("Done!")
    
    return nested_data


def load_dataset(path: str) -> Dict:
    """
    Load a preprocessed dataset from a PyTorch file.
    
    Args:
        path: Path to the .pt file containing the dataset
    
    Returns:
        Dictionary containing the loaded dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    logger.info(f"Loading dataset from {path}...")
    dataset = torch.load(path, map_location='cpu')
    logger.info("Dataset loaded successfully!")
    
    return dataset


def get_dataset_info(dataset: Dict) -> None:
    """
    Print compact information about the dataset structure.
    
    Args:
        dataset: Dataset dictionary
    """
    logger.info("\nDataset Summary:")
    logger.info("=" * 112)
    
    total_samples = 0
    all_machine_types = []
    all_snrs = set()
    
    for machine_type, machine_data in dataset.items():
        all_machine_types.append(machine_type)
        machine_ids = list(machine_data.keys())
        normal_count = 0
        abnormal_count = 0
        type_snrs = set()
        
        for machine_id, snr_data in machine_data.items():
            for snr, normal_abnormal_data in snr_data.items():
                type_snrs.add(snr)
                all_snrs.add(snr)
                if 'normal' in normal_abnormal_data:
                    # normal_abnormal_data['normal'] is now a dict, not a list
                    normal_count += len(normal_abnormal_data['normal'])
                if 'abnormal' in normal_abnormal_data:
                    # normal_abnormal_data['abnormal'] is now a dict, not a list
                    abnormal_count += len(normal_abnormal_data['abnormal'])
        
        total_type_samples = normal_count + abnormal_count
        total_samples += total_type_samples
        
        # Compact format: Machine type | IDs | SNR | Normal | Abnormal | Total
        ids_str = ', '.join(sorted(machine_ids))
        snr_str = ', '.join(sorted(type_snrs))
        logger.info(f"{machine_type:10} | IDs: {ids_str:25} | SNR: {snr_str:12} | "
                   f"Normal: {normal_count:4} | Abnormal: {abnormal_count:4} | Total: {total_type_samples:4}")
    
    logger.info("=" * 112)
    logger.info(f"Total: {len(all_machine_types)} types, {len(all_snrs)} SNR levels, {total_samples} samples")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Preprocess audio dataset with MIMII-like structure for machine learning training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process dataset with default settings
  python prepare_dataset.py data/MIMII
  
  # Process with abnormal samples included
  python prepare_dataset.py data/MIMII --abnormal
  
  # Specify output file and number of jobs
  python prepare_dataset.py data/MIMII --output custom.pt --jobs 8
  
  # Display dataset information
  python prepare_dataset.py data/MIMII --info
        """
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='Path to root directory containing the dataset'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help=f'Path to output file (default: <input_path>/{DEFAULT_DATASET_FILE})'
    )
    
    parser.add_argument(
        '--abnormal', '-a',
        action='store_true',
        help='Include abnormal samples in the dataset'
    )
    
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=None,
        help='Number of parallel processes (default: number of CPU cores - 2)'
    )
    
    parser.add_argument(
        '--info', '-i',
        action='store_true',
        help='Load and display dataset information (requires --output or existing dataset file)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-error output'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if input path exists
    if not os.path.exists(args.input_path):
        logger.error(f"Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    # If --info flag is set, load and display info
    if args.info:
        if args.output is None:
            args.output = os.path.join(args.input_path, DEFAULT_DATASET_FILE)
        
        if not os.path.exists(args.output):
            logger.error(f"Dataset file not found: {args.output}")
            logger.error("Please run preprocessing first.")
            sys.exit(1)
        
        dataset = load_dataset(args.output)
        get_dataset_info(dataset)
        return
    
    # Process the dataset
    try:
        dataset = preprocess_dataset(
            path_in=args.input_path,
            path_out=args.output,
            n_jobs=args.jobs,
            use_abnormal=args.abnormal
        )
        
        # Display dataset info
        get_dataset_info(dataset)
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
