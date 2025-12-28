#!/usr/bin/env python3
"""
Triplet Memory Dataset for Audio Anomaly Detection

This module provides a PyTorch Dataset class for triplet loss training on audio data.
It can be run as a standalone script for testing or imported as a module for training.

Configuration:
    The module supports configuration via environment variables or .env file.
    See env.example for available configuration options.
    If python-dotenv is installed, .env file will be automatically loaded.

Usage as module:
    from triplet_memory_dataset import TripletMemoryDataset
    dataset = TripletMemoryDataset('dataset.pt', samples_per_epoch=5000)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

Usage as script:
    python triplet_memory_dataset.py --dataset dataset.pt [--samples 10] [--plot]
"""

import torch
import torchaudio
import random
import os
import argparse
import logging
import time
from torch.utils.data import Dataset
from typing import Optional, List, Tuple, Union

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
DEFAULT_SAMPLE_RATE = int(os.getenv('DATASET_TARGET_SR', '16000'))
DEFAULT_DURATION_SEC = float(os.getenv('TRIPLET_DURATION_SEC', '1.0'))
DEFAULT_SAMPLES_PER_EPOCH = int(os.getenv('TRIPLET_SAMPLES_PER_EPOCH', '5000'))
DEFAULT_ABNORMAL_RATIO = float(os.getenv('TRIPLET_ABNORMAL_RATIO', '0.5'))

# Default SNR values for MIMII dataset
DEFAULT_SNR_VALUES = ['6db', '0db', '_6db']


class TripletMemoryDataset(Dataset):
    """
    PyTorch Dataset for triplet loss training on audio data.
    
    This dataset generates triplets (anchor, positive, negative) for metric learning.
    Anchor and positive samples come from the same machine ID, while negative samples
    come from different machine IDs or abnormal samples from the same machine.
    
    Args:
        pt_file_path: Path to .pt file containing preprocessed audio data
        duration_sec: Duration of audio chunks in seconds (default: 1.0)
        sample_rate: Sample rate of audio data (default: 16000)
        samples_per_epoch: Number of triplets to generate per epoch (default: 5000)
        anchor_snr: SNR value for anchor samples. Can be a string or None for random selection
        positive_snr: SNR value for positive samples. Can be a string or None for random selection
        negative_snr: SNR value for negative samples. Can be a string or None for random selection
        skip_types: List of machine types to skip (e.g., ['fan'] to exclude fan data)
        abnormal_ratio: Ratio of abnormal samples to use as negatives (default: 0.5)
    """
    
    def __init__(self, pt_file_path: str, duration_sec: float = DEFAULT_DURATION_SEC,
                 sample_rate: int = DEFAULT_SAMPLE_RATE, samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH,
                 anchor_snr: Optional[Union[str, List[str]]] = None,
                 positive_snr: Optional[Union[str, List[str]]] = None,
                 negative_snr: Optional[Union[str, List[str]]] = None,
                 skip_types: List[str] = None,
                 abnormal_ratio: float = DEFAULT_ABNORMAL_RATIO):
        
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration_sec)
        self.samples_per_epoch = samples_per_epoch
        self.abnormal_ratio = abnormal_ratio
        
        if skip_types is None:
            skip_types = []
        
        # Setup SNR selection functions
        # NOTE: Remember to set SNR for anchor, positive and negative
        if anchor_snr is None:
            self._anchor_snr = lambda: random.choice(DEFAULT_SNR_VALUES)
        elif isinstance(anchor_snr, list):
            self._anchor_snr = lambda: random.choice(anchor_snr)
        else:
            self._anchor_snr = lambda: anchor_snr
        
        if positive_snr is None:
            self._positive_snr = lambda: random.choice(DEFAULT_SNR_VALUES)
        elif isinstance(positive_snr, list):
            self._positive_snr = lambda: random.choice(positive_snr)
        else:
            self._positive_snr = lambda: positive_snr
        
        if negative_snr is None:
            self._negative_snr = lambda: random.choice(DEFAULT_SNR_VALUES)
        elif isinstance(negative_snr, list):
            self._negative_snr = lambda: random.choice(negative_snr)
        else:
            self._negative_snr = lambda: negative_snr
        
        logger.info(f"Loading dataset {pt_file_path} into RAM...")
        self.data = torch.load(pt_file_path, weights_only=True)
        logger.info("Dataset loaded! Preparing indices...")
        
        # Check if abnormal samples exist in the dataset
        self.has_abnormal_samples = self._check_abnormal_samples()
        
        # If no abnormal samples and abnormal_ratio > 0, adjust it
        if not self.has_abnormal_samples and abnormal_ratio > 0:
            logger.warning("No abnormal samples found in dataset. Setting abnormal_ratio to 0.0")
            self.abnormal_ratio = 0.0
        
        # Optimization: Create fast access lists
        # self.ids_by_type = { 'fan': ['id_00', 'id_02'], 'pump': [...] }
        self.ids_by_type = {}
        self.machine_types = []
        
        for m_type, id_dict in self.data.items():
            if m_type in skip_types:
                continue
            ids = list(id_dict.keys())
            if len(ids) >= 2:
                self.ids_by_type[m_type] = ids
                self.machine_types.append(m_type)
            else:
                logger.warning(f"Skipping type '{m_type}': insufficient IDs for Triplet Loss (minimum 2 required).")
        
        logger.info(f"Ready machine types: {self.machine_types}")
        
        # Transformation to spectrogram (use static method to ensure consistency)
        self.transform, self.amplitude_to_db = self.create_mel_transform(sample_rate)
    
    def _check_abnormal_samples(self) -> bool:
        """
        Check if abnormal samples exist in the dataset.
        
        Returns:
            True if at least one abnormal sample exists, False otherwise
        """
        for m_type, id_dict in self.data.items():
            for m_id, snr_dict in id_dict.items():
                for snr, normal_dict in snr_dict.items():
                    if isinstance(normal_dict, dict) and 'abnormal' in normal_dict:
                        abnormal_samples = normal_dict['abnormal']
                        if isinstance(abnormal_samples, dict) and len(abnormal_samples) > 0:
                            return True
                        elif isinstance(abnormal_samples, list) and len(abnormal_samples) > 0:
                            return True
        return False
    
    def get_random_crop(self, full_wav_int16: torch.Tensor) -> torch.Tensor:
        """
        Extract a random crop from the full waveform.
        
        Args:
            full_wav_int16: Full waveform tensor in int16 format [channels, samples]
        
        Returns:
            Cropped waveform tensor in float32 format normalized to [-1.0, 1.0]
        """
        total_len = full_wav_int16.shape[1]
        
        # Random crop
        if total_len > self.target_len:
            start = random.randint(0, total_len - self.target_len)
            crop_int16 = full_wav_int16[:, start : start + self.target_len]
        else:
            # Padding if recording is short
            padding = self.target_len - total_len
            crop_int16 = torch.nn.functional.pad(full_wav_int16, (0, padding))
        
        # Convert int16 -> float32
        # Divide by 32767 to get range [-1.0, 1.0]
        return crop_int16.float() / 32767.0
    
    def get_random_sample(self, m_type: str, m_id: str, snr: str, normal: str = 'normal') -> Tuple[torch.Tensor, int]:
        """
        Get a random sample from the dataset.
        
        Args:
            m_type: Machine type (e.g., 'fan', 'pump')
            m_id: Machine ID (e.g., 'id_00', 'id_02')
            snr: SNR level (e.g., '6db', '0db', '_6db')
            normal: 'normal' or 'abnormal'
        
        Returns:
            Tuple of (waveform tensor, sample_id)
        """
        # Check if abnormal samples exist for this specific machine/SNR
        if normal != 'normal' and not self.has_abnormal_samples:
            raise KeyError(f"No abnormal samples available for {m_type}[{m_id}] at SNR {snr}")
        
        # Get list of tensors (int16) for this ID
        # This works instantly (hash table)
        recordings = self.data[m_type][m_id][snr][normal]
        
        # Select random recording
        sample_id = random.randrange(len(recordings))
        return recordings[sample_id], sample_id
    
    def get_sample(self, m_type: str, m_id: str, snr: str, normal: str = 'normal',
                   sample_id: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        Get a specific sample or random sample from the dataset.
        
        Args:
            m_type: Machine type
            m_id: Machine ID
            snr: SNR level
            normal: 'normal' or 'abnormal'
            sample_id: Specific sample ID, or None for random selection
        
        Returns:
            Tuple of (waveform tensor, sample_id)
        """
        if normal != 'normal' and not self.has_abnormal_samples:
            raise KeyError(f"No abnormal samples available for {m_type}[{m_id}] at SNR {snr}")

        if sample_id is None:
            return self.get_random_sample(m_type, m_id, snr, normal)
        else:
            return self.data[m_type][m_id][snr][normal][sample_id], sample_id
    
    def get_triplet_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str, str]:
        """
        Generate a triplet sample (anchor, positive, negative).
        
        Returns:
            Tuple of (anchor_wav, positive_wav, negative_wav, anchor_name, positive_name, negative_name)
        """
        # Select machine type (fan, valve...)
        m_type = random.choice(self.machine_types)
        
        # Get available IDs for this type (from cache)
        available_ids = self.ids_by_type[m_type]
        
        # Anchor and Negative: Random IDs from same type
        anchor_id, neg_id = random.sample(available_ids, k=2)
        positive_normal = 'normal'
        
        # Determine negative sample type
        # Only use abnormal if ratio > 0 AND abnormal samples exist AND random chance
        use_abnormal = (self.abnormal_ratio > 0 and 
                       self.has_abnormal_samples and 
                       random.random() < self.abnormal_ratio)
        
        if use_abnormal:
            negative_normal = 'abnormal'
            neg_id = anchor_id
        else:
            negative_normal = 'normal'
        
        anchor_wav, anchor_sample_id = self.get_random_sample(
            m_type, anchor_id, self._anchor_snr(), normal=positive_normal
        )
        positive_wav, positive_sample_id = self.get_random_sample(
            m_type, anchor_id, self._positive_snr(), normal=positive_normal
        )
        negative_wav, negative_sample_id = self.get_random_sample(
            m_type, neg_id, self._negative_snr(), normal=negative_normal
        )
        
        anchor_name = f"{m_type}[{anchor_id}]-{self._anchor_snr()}-{positive_normal}[{anchor_sample_id}]"
        positive_name = f"{m_type}[{anchor_id}]-{self._positive_snr()}-{positive_normal}[{positive_sample_id}]"
        negative_name = f"{m_type}[{neg_id}]-{self._negative_snr()}-{negative_normal}[{negative_sample_id}]"
        
        return anchor_wav, positive_wav, negative_wav, anchor_name, positive_name, negative_name
    
    def get_mel_spec(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram.
        
        Args:
            wav: Waveform tensor [channels, samples]
        
        Returns:
            Mel spectrogram tensor [channels, n_mels, time_frames]
        """
        spec = self.transform(wav)
        return self.amplitude_to_db(spec)
    
    @staticmethod
    def create_mel_transform(sample_rate: int = DEFAULT_SAMPLE_RATE):
        """
        Create mel spectrogram transform (same as used in dataset).
        
        Args:
            sample_rate: Sample rate (default: 16000)
        
        Returns:
            Tuple of (transform, amplitude_to_db)
        """
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=64,
            n_fft=1024,
            hop_length=512,
            power=2.0
        )
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        return transform, amplitude_to_db
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a triplet sample for training.
        
        Args:
            idx: Index (not used, samples are randomly generated)
        
        Returns:
            Tuple of (anchor_spec, positive_spec, negative_spec)
        """
        # Triplet mining logic
        anchor_wav, positive_wav, negative_wav, anchor_name, positive_name, negative_name = self.get_triplet_sample()
        
        anchor_wav_crop = self.get_random_crop(anchor_wav)
        positive_wav_crop = self.get_random_crop(positive_wav)
        negative_wav_crop = self.get_random_crop(negative_wav)
        
        # Create spectrograms
        # MelSpectrogram expects [channel, time] or [time]
        # Our input is [1, 16000]. Output will be [1, n_mels, time_frames]
        a_spec = self.get_mel_spec(anchor_wav_crop)
        p_spec = self.get_mel_spec(positive_wav_crop)
        n_spec = self.get_mel_spec(negative_wav_crop)
        
        return a_spec, p_spec, n_spec
    
    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self.samples_per_epoch


def benchmark_performance(dataset: TripletMemoryDataset, num_samples: int = 5000, 
                          warmup_samples: int = 100) -> dict:
    """
    Benchmark the performance of triplet generation.
    
    Args:
        dataset: TripletMemoryDataset instance
        num_samples: Number of triplets to generate for benchmarking
        warmup_samples: Number of warmup samples to skip (for cache warming)
    
    Returns:
        Dictionary with performance metrics:
        - total_time: Total time in seconds
        - samples_per_second: Throughput
        - avg_time_per_sample: Average time per triplet in milliseconds
        - min_time: Minimum time per sample in milliseconds
        - max_time: Maximum time per sample in milliseconds
    """
    logger.info(f"Starting performance benchmark...")
    logger.info(f"Warmup samples: {warmup_samples}")
    logger.info(f"Benchmark samples: {num_samples}")
    
    # Warmup phase (to warm up caches, JIT compilation, etc.)
    logger.info("Warmup phase...")
    warmup_start = time.time()
    for i in range(warmup_samples):
        _ = dataset[i]
    warmup_time = time.time() - warmup_start
    logger.info(f"Warmup completed in {warmup_time:.3f}s ({warmup_samples/warmup_time:.1f} samples/s)")
    
    # Benchmark phase
    logger.info("Benchmark phase...")
    times = []
    total_start = time.time()
    
    for i in range(num_samples):
        sample_start = time.time()
        _ = dataset[i]
        sample_time = (time.time() - sample_start) * 1000  # Convert to milliseconds
        times.append(sample_time)
        
        # Log progress every 10%
        if (i + 1) % (num_samples // 10) == 0:
            progress = (i + 1) / num_samples * 100
            elapsed = time.time() - total_start
            rate = (i + 1) / elapsed
            logger.info(f"Progress: {progress:.0f}% ({i + 1}/{num_samples}) - "
                       f"Rate: {rate:.1f} samples/s - "
                       f"ETA: {(num_samples - i - 1) / rate:.1f}s")
    
    total_time = time.time() - total_start
    
    # Calculate statistics
    times_array = torch.tensor(times)
    avg_time = times_array.mean().item()
    min_time = times_array.min().item()
    max_time = times_array.max().item()
    median_time = times_array.median().item()
    std_time = times_array.std().item()
    
    samples_per_second = num_samples / total_time
    
    results = {
        'total_time': total_time,
        'samples_per_second': samples_per_second,
        'avg_time_per_sample': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'median_time': median_time,
        'std_time': std_time,
        'warmup_time': warmup_time,
        'num_samples': num_samples
    }
    
    # Print results
    logger.info("=" * 60)
    logger.info("Performance Benchmark Results")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.3f}s")
    logger.info(f"Throughput: {samples_per_second:.2f} samples/second")
    logger.info(f"Average time per sample: {avg_time:.3f} ms")
    logger.info(f"Median time per sample: {median_time:.3f} ms")
    logger.info(f"Min time per sample: {min_time:.3f} ms")
    logger.info(f"Max time per sample: {max_time:.3f} ms")
    logger.info(f"Std deviation: {std_time:.3f} ms")
    logger.info(f"Warmup time: {warmup_time:.3f}s")
    logger.info("=" * 60)
    
    return results


def plot_spectrograms(a: torch.Tensor, p: torch.Tensor, n: torch.Tensor):
    """
    Plot three spectrograms side by side for visualization.
    
    Args:
        a: Anchor spectrogram
        p: Positive spectrogram
        n: Negative spectrogram
    """
    try:
        import matplotlib.pyplot as plt
        
        logger.info("Getting one batch...")
        logger.info(f"Spectrogram shape: {a.shape}")
        logger.info(f"Min value: {a.min()}, Max value: {a.max()}")
        
        # Visualization
        plt.figure(figsize=(10, 8))
        
        ax1 = plt.subplot(3, 1, 1)
        im1 = plt.imshow(a.squeeze().numpy(), aspect='auto', origin='lower')
        plt.title("Anchor (Machine A)")
        cbar1 = plt.colorbar(im1, ax=ax1, format='%+2.0f dB')
        cbar1.set_label('Intensity (dB)')
        
        ax2 = plt.subplot(3, 1, 2)
        im2 = plt.imshow(p.squeeze().numpy(), aspect='auto', origin='lower')
        plt.title("Positive (Machine A - same ID)")
        cbar2 = plt.colorbar(im2, ax=ax2, format='%+2.0f dB')
        cbar2.set_label('Intensity (dB)')
        
        ax3 = plt.subplot(3, 1, 3)
        im3 = plt.imshow(n.squeeze().numpy(), aspect='auto', origin='lower')
        plt.title("Negative (Machine B - diff ID)")
        cbar3 = plt.colorbar(im3, ax=ax3, format='%+2.0f dB')
        cbar3.set_label('Intensity (dB)')
        
        plt.tight_layout()
        plt.show()
        logger.info("✅ Test passed successfully!")
        
    except ImportError:
        logger.error("matplotlib not available. Install it to use plotting functionality.")
    except Exception as e:
        logger.error(f"Error plotting spectrograms: {e}", exc_info=True)


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Test TripletMemoryDataset with visualization'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=DEFAULT_DATASET_FILE,
        help=f'Path to dataset .pt file (default: {DEFAULT_DATASET_FILE})'
    )
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=10,
        help='Number of samples per epoch (default: 10)'
    )
    parser.add_argument(
        '--plot', '-p',
        action='store_true',
        help='Plot spectrograms for visualization'
    )
    parser.add_argument(
        '--duration', '-t',
        type=float,
        default=DEFAULT_DURATION_SEC,
        help=f'Audio duration in seconds (default: {DEFAULT_DURATION_SEC})'
    )
    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run performance benchmark (5000 samples by default)'
    )
    parser.add_argument(
        '--benchmark-samples',
        type=int,
        default=5000,
        help='Number of samples for benchmark (default: 5000)'
    )
    parser.add_argument(
        '--warmup-samples',
        type=int,
        default=100,
        help='Number of warmup samples before benchmark (default: 100)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        logger.error("Please run preprocessing first or check the path.")
        return 1
    
    try:
        logger.info(f"Loading dataset from {args.dataset}...")
        dataset = TripletMemoryDataset(
            args.dataset,
            samples_per_epoch=args.samples,
            duration_sec=args.duration
        )
        
        logger.info(f"Dataset loaded successfully! Length: {len(dataset)}")
        
        # Run benchmark if requested
        if args.benchmark:
            benchmark_performance(dataset, args.benchmark_samples, args.warmup_samples)
            return 0
        
        # Test getting a triplet sample
        logger.info("Testing triplet sample generation...")
        anchor_wav, positive_wav, negative_wav, anchor_name, positive_name, negative_name = dataset.get_triplet_sample()
        logger.info(f"Anchor: {anchor_name}")
        logger.info(f"Positive: {positive_name}")
        logger.info(f"Negative: {negative_name}")
        
        # Test getting a batch item
        logger.info("Testing __getitem__...")
        a_spec, p_spec, n_spec = dataset[0]
        logger.info(f"Anchor spec shape: {a_spec.shape}")
        logger.info(f"Positive spec shape: {p_spec.shape}")
        logger.info(f"Negative spec shape: {n_spec.shape}")
        
        if args.plot:
            plot_spectrograms(a_spec, p_spec, n_spec)
        
        logger.info("✅ All tests passed!")
        return 0
        
    except FileNotFoundError:
        logger.error(f"File not found: {args.dataset}")
        logger.error("Please check the path.")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
