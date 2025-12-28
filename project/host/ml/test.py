#!/usr/bin/env python3
"""
Model Testing and Validation Module for Audio Anomaly Detection

This module provides comprehensive model testing and validation functionality
with MLflow integration. It can be run as a standalone script or imported as a module.

Configuration:
    The module supports configuration via environment variables or .env file.
    See env.example for available configuration options.
    If python-dotenv is installed, .env file will be automatically loaded.

Usage as module:
    from test import validate_model, run_full_validation
    results = validate_model(model, dataset, device, 'fan', rms_threshold=0.005)

Usage as script:
    # Run validation (normal mode)
    python test.py --model runs:/<run_id>/models/model.pt --dataset dataset.pt
    
    # Quick infrastructure test (verify module setup)
    python test.py --test
    
    # Two-file WAV testing mode
    python test.py --model runs:/<run_id>/models/model.pt --normal normal.wav --anomaly anomaly.wav
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import json
import argparse
import logging
import tempfile
from typing import Optional, Dict, Any, List
from tqdm import tqdm

# Configure logging (must be before MinIO config to use logger)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import torchaudio (required for WAV file testing)
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    logger.warning("torchaudio not available. WAV file testing will be disabled.")

# Try to import soundfile (required for WAV file loading)
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    logger.warning("soundfile not available. WAV file testing will be disabled.")

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure MinIO/S3 settings for MLflow (if using MinIO instead of S3)
# These should be set before importing mlflow or calling mlflow functions
MLFLOW_S3_ENDPOINT_URL = os.getenv('MLFLOW_S3_ENDPOINT_URL', None)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', None)
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)
MLFLOW_S3_IGNORE_TLS = os.getenv('MLFLOW_S3_IGNORE_TLS', 'false').lower() == 'true'

if MLFLOW_S3_ENDPOINT_URL:
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = MLFLOW_S3_ENDPOINT_URL
    logger.debug(f"MLflow S3 endpoint configured: {MLFLOW_S3_ENDPOINT_URL}")

if AWS_ACCESS_KEY_ID:
    os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
    logger.debug("AWS access key ID configured")

if AWS_SECRET_ACCESS_KEY:
    os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
    logger.debug("AWS secret access key configured")

if MLFLOW_S3_IGNORE_TLS:
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    logger.debug("MLflow S3 TLS verification disabled")

# Try to import mlflow (optional dependency)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")

# Try to import matplotlib (optional dependency)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Plotting will be disabled.")

# Import local modules
# Support both absolute and relative imports
try:
    from ml.model import TinyAudioCNN
    from ml.triplet_memory_dataset import TripletMemoryDataset
except ImportError:
    # Fallback to relative imports (when running as script from ml/ directory)
    from model import TinyAudioCNN
    from triplet_memory_dataset import TripletMemoryDataset

# Constants (can be overridden via environment variables)
DEFAULT_DATASET_FILE = os.getenv('DEFAULT_DATASET_FILE', 'dataset.pt')
DEFAULT_EMBEDDING_DIM = int(os.getenv('MODEL_EMBEDDING_DIM', '64'))
DEFAULT_RMS_THRESHOLD = float(os.getenv('TEST_RMS_THRESHOLD', '0.005'))
DEFAULT_MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
DEFAULT_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Model_testing_validation')

# Default SNR values for MIMII dataset
DEFAULT_SNR_VALUES = ['6db', '0db', '_6db']


def calculate_rms(waveform_tensor: torch.Tensor) -> float:
    """
    Calculate Root Mean Square (RMS) amplitude of a waveform tensor.
    
    Args:
        waveform_tensor: Input tensor of shape [1, 16000] or [16000]
    
    Returns:
        RMS value as float
    """
    return torch.sqrt(torch.mean(waveform_tensor**2)).item()


def get_embedding(model: nn.Module, batch_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Get embeddings from model for a batch of spectrograms.
    
    Args:
        model: TinyAudioCNN model instance
        batch_tensor: Input tensor of shape [Batch, 1, mel_bins, time_frames] or [Batch, mel_bins, time_frames]
        device: Target device (torch.device)
    
    Returns:
        Embedding tensor of shape [Batch, embedding_dim]
    """
    with torch.no_grad():
        if len(batch_tensor.shape) == 3:
            batch_tensor = batch_tensor.unsqueeze(1)
        return model(batch_tensor.to(device))


def load_model_mlflow(model_path: str, device: torch.device, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> nn.Module:
    """
    Load model from local file or MLflow.
    
    Args:
        model_path: Path to model. Can be:
            - Local path: "/path/to/model.pth"
            - MLflow URI: "runs:/<run_id>/models/<filename>"
            - MLflow run_id: "<run_id>" (will load best_model.pth from artifacts)
        device: Target device (torch.device)
        embedding_dim: Size of embedding vector (default: 64)
    
    Returns:
        Loaded TinyAudioCNN model in eval mode
    """
    logger.info(f"Loading model from {model_path}...")
    
    # Check if this is an MLflow URI
    if model_path.startswith("runs:/"):
        # MLflow URI format: runs:/<run_id>/models/<filename>
        logger.info("Loading model from MLflow...")
        try:
            # Download artifact from MLflow
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_path)
            logger.info(f"Model downloaded from MLflow: {local_path}")
            model_path = local_path
        except Exception as e:
            logger.error(f"Error downloading from MLflow: {e}")
            raise
    elif len(model_path) == 32 and all(c in '0123456789abcdef' for c in model_path.lower()):
        # Possibly a run_id (32 hex characters)
        logger.info(f"Loading model from MLflow run_id: {model_path}")
        try:
            # Look for best_model in artifacts
            artifact_uri = f"runs:/{model_path}/models"
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri)
            # Look for .pth file in downloaded directory
            if os.path.isdir(local_path):
                pth_files = [f for f in os.listdir(local_path) if f.endswith('.pth')]
                if pth_files:
                    # Take first .pth file or look for best_model
                    best_file = next((f for f in pth_files if 'best' in f.lower()), pth_files[0])
                    model_path = os.path.join(local_path, best_file)
                else:
                    raise FileNotFoundError(f"No .pth files found in {local_path}")
            logger.info(f"Model downloaded from MLflow: {model_path}")
        except Exception as e:
            logger.error(f"Error downloading from MLflow: {e}")
            raise
    
    # Load model
    model = TinyAudioCNN(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded successfully!")
    return model


def validate_model(
    model: nn.Module,
    dataset: TripletMemoryDataset,
    device: torch.device,
    target_type: str,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    snr: str = '6db',
    mlflow_run_id: Optional[str] = None,
    fixed_target_id: Optional[str] = None,
    silent: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Validate model with automatic logging of results to MLflow.
    
    Args:
        model: Model to validate
        dataset: Dataset for validation
        device: Device (torch.device)
        target_type: Machine type ('fan', 'pump', 'slider', 'valve')
        rms_threshold: RMS threshold for silence filtering
        snr: SNR level ('6db', '0db', '_6db')
        mlflow_run_id: ID of existing MLflow run for logging (None = use active run)
        fixed_target_id: Fixed ID for testing (None = random)
        silent: If True, suppress output and plots
    
    Returns:
        Dictionary with validation results or None if validation failed
    """
    if fixed_target_id is None:
        target_ids = dataset.ids_by_type[target_type]
        if len(target_ids) < 2:
            logger.error(f"Not enough IDs for type {target_type}")
            return None
        target_id, other_id = random.sample(target_ids, k=2)
    else:
        target_id = fixed_target_id
        target_ids = dataset.ids_by_type[target_type]
        other_id = random.choice(target_ids)
        while other_id == target_id:
            other_id = random.choice(target_ids)
    
    # Connect to MLflow run
    # If mlflow_run_id is specified, connect to existing run
    # If not - use active run or work without logging
    use_mlflow = False
    run_id_to_use = None
    
    if mlflow_run_id is not None:
        # Connect to existing run
        if not (isinstance(mlflow_run_id, bool) and mlflow_run_id == False):
            use_mlflow = True
            run_id_to_use = mlflow_run_id
            if not silent:
                logger.info(f"Connecting to MLflow run: {mlflow_run_id}")
    else:
        # Check if there is an active run
        if MLFLOW_AVAILABLE:
            active_run = mlflow.active_run()
            if active_run:
                run_id_to_use = active_run.info.run_id
                use_mlflow = True
                if not silent:
                    logger.info(f"Using active MLflow run: {run_id_to_use}")
            else:
                if not silent:
                    logger.warning("No active MLflow run and mlflow_run_id not specified. Results will not be logged.")
    
    if not silent:
        logger.info(f"\n🧪 EXPERIMENT: {target_type.upper()} ({snr})")
        logger.info(f"🟢 Target: {target_id} | 🔴 Other: {other_id}")
    
    # --- 1. CALIBRATION WITH RMS FILTER ---
    # Calibration happens on only one sample. Extract X random chunks of 1 second each
    calibration_mels = []
    # Get base sample for calibration
    base_wav, base_wav_id = dataset.get_sample(target_type, target_id, snr, normal='normal')
    attempts = 0
    while len(calibration_mels) < 50 and attempts < 500:
        attempts += 1
        w = dataset.get_random_crop(base_wav)
        
        # RMS check
        rms = calculate_rms(w)
        if rms < rms_threshold:
            continue
        
        s = dataset.get_mel_spec(w)
        calibration_mels.append(s)
    
    if len(calibration_mels) < 10:
        logger.warning("Failed to find enough loud samples for calibration!")
        return None
    
    calib_batch = torch.stack(calibration_mels)
    calib_vecs = get_embedding(model, calib_batch, device)
    golden_vector = torch.mean(calib_vecs, dim=0, keepdim=True)
    dists = torch.norm(calib_vecs - golden_vector, dim=1).cpu().numpy()
    
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    threshold = mean_dist + 3 * std_dist
    
    if not silent:
        logger.info(f"📊 Calibration (RMS > {rms_threshold}) - {target_type}-{target_id}-{snr}-normal[{base_wav_id}]:")
        logger.info(f"   Mean: {mean_dist:.4f} | Std: {std_dist:.4f} | Threshold: {threshold:.4f}")
    
    # --- 2. TESTING ---
    results = {'normal': [], 'anomaly': []}
    
    def run_test_loop(t_id, normal, sample_id, result_list):
        count = 0
        local_attempts = 0
        test_wav, test_wav_id = dataset.get_sample(target_type, t_id, snr, normal=normal, sample_id=sample_id)
        while count < 50 and local_attempts < 500:
            local_attempts += 1
            w = dataset.get_random_crop(test_wav)
            
            if calculate_rms(w) < rms_threshold:
                continue
            
            s = dataset.get_mel_spec(w)
            v = get_embedding(model, s.unsqueeze(0), device)
            dist = torch.norm(v - golden_vector).item()
            result_list.append(dist)
            count += 1
        return count
    
    n_norm = run_test_loop(target_id, 'normal', base_wav_id, results['normal'])
    n_anom = run_test_loop(target_id, 'abnormal', None, results['anomaly'])
    
    if not silent:
        logger.info(f"🔍 Test (found loud samples): Normal={n_norm}, Anomaly={n_anom}")
    
    # Statistics
    validation_results = None
    if len(results['normal']) > 0 and len(results['anomaly']) > 0:
        false_positives = sum(d > threshold for d in results['normal'])
        false_negatives = sum(d <= threshold for d in results['anomaly'])
        
        avg_dist_normal = np.mean(results['normal'])
        avg_dist_anomaly = np.mean(results['anomaly'])
        
        if not silent:
            logger.info(f"   AVG Dist Normal:  {avg_dist_normal:.4f}")
            logger.info(f"   AVG Dist Anomaly: {avg_dist_anomaly:.4f}")
            logger.info(f"   ⚠️ False Positives: {false_positives}/{n_norm}")
            logger.info(f"   ⚠️ Missed Anomalies: {false_negatives}/{n_anom}")
        
        # Prepare results for MLflow
        validation_results = {
            'target_type': target_type,
            'snr': snr,
            'target_id': target_id,
            'other_id': other_id,
            'threshold': float(threshold),
            'mean_dist_normal': float(mean_dist),
            'std_dist_normal': float(std_dist),
            'avg_dist_normal': float(avg_dist_normal),
            'avg_dist_anomaly': float(avg_dist_anomaly),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'total_normal': int(n_norm),
            'total_anomaly': int(n_anom),
            'false_positive_rate': float(false_positives / n_norm) if n_norm > 0 else 0.0,
            'false_negative_rate': float(false_negatives / n_anom) if n_anom > 0 else 0.0,
        }
        
        # Logging results to MLflow
        if use_mlflow and run_id_to_use and MLFLOW_AVAILABLE:
            try:
                # Check if active run matches the required one
                active_run = mlflow.active_run()
                need_to_start_run = True
                
                if active_run and active_run.info.run_id == run_id_to_use:
                    # Active run is already correct
                    need_to_start_run = False
                    if not silent:
                        logger.info(f"Using active MLflow run: {run_id_to_use}")
                
                # Connect to run (if needed)
                if need_to_start_run:
                    # Use start_run to connect to existing run
                    mlflow.start_run(run_id=run_id_to_use)
                    if not silent:
                        logger.info(f"Connected to MLflow run: {run_id_to_use}")
                
                # Log metrics
                mlflow.log_metrics({
                    f'validation_{target_type}_{snr}_threshold': threshold,
                    f'validation_{target_type}_{snr}_avg_dist_normal': avg_dist_normal,
                    f'validation_{target_type}_{snr}_avg_dist_anomaly': avg_dist_anomaly,
                    f'validation_{target_type}_{snr}_false_positive_rate': validation_results['false_positive_rate'],
                    f'validation_{target_type}_{snr}_false_negative_rate': validation_results['false_negative_rate'],
                    f'validation_{target_type}_{snr}_false_positives': false_positives,
                    f'validation_{target_type}_{snr}_false_negatives': false_negatives,
                })
                
                # Log validation parameters
                mlflow.log_params({
                    f'validation_{target_type}_{snr}_target_id': target_id,
                    f'validation_{target_type}_{snr}_other_id': other_id,
                    f'validation_{target_type}_{snr}_calibration_samples': len(calibration_mels),
                })
                
                # Save results as JSON artifact
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(validation_results, f, indent=2)
                    temp_path = f.name
                
                mlflow.log_artifact(temp_path, f"validation_results/{target_type}_{snr}")
                os.unlink(temp_path)
                
                # Save histogram as artifact
                if MATPLOTLIB_AVAILABLE:
                    plt.figure(figsize=(10, 4))
                    plt.hist(results['normal'], bins=15, alpha=0.7, color='green', label='Normal')
                    plt.hist(results['anomaly'], bins=15, alpha=0.7, color='red', label='Anomaly')
                    plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
                    plt.title(f"{target_type} ({snr}) - RMS Filtered")
                    plt.xlabel('Distance')
                    plt.ylabel('Frequency')
                    plt.legend()
                    
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                        plot_path = f.name
                    plt.savefig(plot_path)
                    plt.close()
                    
                    mlflow.log_artifact(plot_path, f"validation_plots/{target_type}_{snr}")
                    os.unlink(plot_path)
                
                if not silent:
                    logger.info(f"\n📊 Validation results saved to MLflow run: {run_id_to_use}")
                
                # Close run only if we opened it
                if need_to_start_run:
                    mlflow.end_run()
                    
            except Exception as e:
                logger.warning(f"Failed to save results to MLflow: {e}")
                if not silent:
                    import traceback
                    traceback.print_exc()
        
        # Visualization (always show if not silent)
        if not silent and MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(10, 4))
            plt.hist(results['normal'], bins=15, alpha=0.7, color='green', label='Normal')
            plt.hist(results['anomaly'], bins=15, alpha=0.7, color='red', label='Anomaly')
            plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
            plt.title(f"{target_type}")
            plt.legend()
            plt.show()
    else:
        logger.warning("Insufficient data after filtering.")
    
    return validation_results


def run_full_validation(
    model_path: str,
    dataset_file: str,
    device: Optional[torch.device] = None,
    snr: str = '6db',
    skip_types: List[str] = None,
    num_iterations: int = 50,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    mlflow_run_id: Optional[str] = None
) -> Dict[str, float]:
    """
    Run full validation across all machine types with multiple iterations.
    
    Args:
        model_path: Path to model (local or MLflow URI)
        dataset_file: Path to dataset .pt file
        device: Device to use (None = auto-detect)
        snr: SNR level to test ('6db', '0db', '_6db')
        skip_types: List of machine types to skip
        num_iterations: Number of validation iterations per machine type
        rms_threshold: RMS threshold for silence filtering
        embedding_dim: Embedding dimension of the model
        mlflow_run_id: MLflow run ID for logging results
    
    Returns:
        Dictionary with average false_positive_rate and false_negative_rate
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    if skip_types is None:
        skip_types = []
    
    logger.info(f"Loading dataset from {dataset_file}...")
    dataset = TripletMemoryDataset(dataset_file, samples_per_epoch=100, skip_types=skip_types)
    
    logger.info(f"Loading model from {model_path}...")
    model = load_model_mlflow(model_path, device, embedding_dim=embedding_dim)
    
    # Extract MLflow run_id from model_path if it's an MLflow URI
    if mlflow_run_id is None:
        if model_path.startswith("runs:/"):
            # Extract run_id from URI: runs:/<run_id>/models/...
            parts = model_path.split('/')
            if len(parts) >= 2:
                mlflow_run_id = parts[1]
        elif len(model_path) == 32 and all(c in '0123456789abcdef' for c in model_path.lower()):
            mlflow_run_id = model_path
    
    total_results = {
        'false_positive_rate': 0.0,
        'false_negative_rate': 0.0,
    }
    
    # Get available machine types from dataset (dynamically determined from dataset)
    # Machine types are determined by what's available in the dataset minus skip_types
    available_types = [t for t in dataset.ids_by_type.keys() if t not in skip_types]
    
    if not available_types:
        logger.error("No available machine types to validate!")
        logger.error(f"Dataset contains types: {list(dataset.ids_by_type.keys())}")
        logger.error(f"Skipped types: {skip_types}")
        return total_results
    
    logger.info(f"Available machine types for validation: {available_types}")
    
    count = 0
    silent = False
    current_mlflow_run_id = mlflow_run_id
    
    logger.info(f"Starting full validation: {num_iterations} iterations × {len(available_types)} types")
    
    for iteration in tqdm(range(num_iterations), desc="Validation iterations"):
        for target_type in available_types:
            count += 1
            validation_results = validate_model(
                model, dataset, device, target_type=target_type,
                rms_threshold=rms_threshold, snr=snr,
                mlflow_run_id=current_mlflow_run_id, silent=silent
            )
            
            if validation_results:
                total_results['false_positive_rate'] += validation_results['false_positive_rate']
                total_results['false_negative_rate'] += validation_results['false_negative_rate']
        
        # After first iteration, suppress output and don't reconnect to MLflow
        silent = True
        current_mlflow_run_id = False
    
    if count > 0:
        total_results['false_positive_rate'] /= count
        total_results['false_negative_rate'] /= count
    
    # Log total results to MLflow
    if mlflow_run_id and MLFLOW_AVAILABLE:
        try:
            logger.info(f"Connecting to MLflow run_id: {mlflow_run_id}")
            active_run = mlflow.active_run()
            started_run = False
            
            if not active_run:
                logger.info(f"Active run not found, starting new run with id {mlflow_run_id}")
                mlflow.start_run(run_id=mlflow_run_id)
                started_run = True
            elif active_run.info.run_id != mlflow_run_id:
                logger.info(f"Run ID changed from {active_run.info.run_id} to {mlflow_run_id}")
                mlflow.end_run()
                mlflow.start_run(run_id=mlflow_run_id)
                started_run = True
            
            # Log metrics
            mlflow.log_metrics({
                'total_false_positive_rate': total_results['false_positive_rate'],
                'total_false_negative_rate': total_results['false_negative_rate'],
            })
            
            if started_run:
                mlflow.end_run()
        except Exception as e:
            logger.warning(f"Failed to log total results to MLflow: {e}")
    
    logger.info(f"Total False Positive Rate: {total_results['false_positive_rate']:.4f}")
    logger.info(f"Total False Negative Rate: {total_results['false_negative_rate']:.4f}")
    
    return total_results


def test_two_wav_files(
    model_path: str,
    normal_wav_path: str,
    anomaly_wav_path: str,
    device: Optional[torch.device] = None,
    sample_rate: int = 16000,
    duration_sec: float = 1.0,
    calibration_chunks: int = 50,
    test_chunks: int = 50,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    show_plot: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Test model on two WAV files: one normal (good) and one anomaly (bad).
    
    Logic:
    - First half of normal file is used for calibration (golden vector)
    - Second half of normal file and entire anomaly file are used for testing
    
    Args:
        model_path: Path to model (local path, MLflow URI, or MLflow run_id)
        normal_wav_path: Path to normal (good) WAV file
        anomaly_wav_path: Path to anomaly (bad) WAV file
        device: Device to use (None = auto-detect)
        sample_rate: Sample rate (default: 16000)
        duration_sec: Duration of chunks in seconds (default: 1.0)
        calibration_chunks: Number of chunks for calibration (default: 50)
        test_chunks: Number of chunks for testing (default: 50)
        rms_threshold: RMS threshold for silence filtering (default: 0.005)
        embedding_dim: Embedding dimension (default: 64)
        show_plot: Whether to show visualization plot (default: True)
    
    Returns:
        Dictionary with test results or None if test failed
    """
    if not TORCHAUDIO_AVAILABLE:
        logger.error("torchaudio is required for WAV file testing")
        return None
    
    if not SOUNDFILE_AVAILABLE:
        logger.error("soundfile is required for WAV file testing")
        return None
    
    # Determine device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    model = load_model_mlflow(model_path, device, embedding_dim=embedding_dim)
    model = model.to(device)
    model.eval()
    
    # Create transforms using dataset's static method
    transform, amplitude_to_db = TripletMemoryDataset.create_mel_transform(sample_rate)
    target_len = int(sample_rate * duration_sec)
    
    # Import load_and_process_wav_file from prepare_dataset module
    try:
        from ml.prepare_dataset import load_and_process_wav_file
    except ImportError:
        # Fallback to relative import (when running as script from ml/ directory)
        from prepare_dataset import load_and_process_wav_file
    
    # Load and process WAV files using prepare_dataset function (exact same logic as dataloader)
    logger.info(f"Loading normal WAV file: {normal_wav_path}")
    try:
        normal_wav_int16 = load_and_process_wav_file(
            normal_wav_path, 
            sample_rate=sample_rate
        )
        logger.info(f"Normal file loaded: {normal_wav_int16.shape[1] / sample_rate:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load normal WAV file: {e}")
        return None
    
    # Load and process anomaly WAV file
    logger.info(f"Loading anomaly WAV file: {anomaly_wav_path}")
    try:
        anomaly_wav_int16 = load_and_process_wav_file(
            anomaly_wav_path,
            sample_rate=sample_rate
        )
        logger.info(f"Anomaly file loaded: {anomaly_wav_int16.shape[1] / sample_rate:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to load anomaly WAV file: {e}")
        return None
    
    # Split normal file in half (using int16 format as in dataset)
    total_len = normal_wav_int16.shape[1]
    half_len = total_len // 2
    
    first_half_int16 = normal_wav_int16[:, :half_len]
    second_half_int16 = normal_wav_int16[:, half_len:]
    
    def extract_random_chunks(wav_int16: torch.Tensor, num_chunks: int, min_rms: Optional[float] = None) -> List[torch.Tensor]:
        """
        Extract random chunks from waveform using dataset's logic.
        Replicates exact behavior of TripletMemoryDataset.get_random_crop and get_mel_spec.
        
        Args:
            wav_int16: Waveform tensor in int16 format [1, samples]
            num_chunks: Number of chunks to extract
            min_rms: Minimum RMS for filtering (None = no filtering)
        
        Returns:
            List of processed spectrograms [num_chunks, 1, 64, 32]
        """
        chunks = []
        attempts = 0
        max_attempts = num_chunks * 10
        
        while len(chunks) < num_chunks and attempts < max_attempts:
            attempts += 1
            
            # Replicate dataset's get_random_crop logic
            total_len = wav_int16.shape[1]
            if total_len > target_len:
                start = random.randint(0, total_len - target_len)
                crop_int16 = wav_int16[:, start : start + target_len]
            else:
                # Padding if recording is short
                padding = target_len - total_len
                crop_int16 = F.pad(wav_int16, (0, padding))
            
            # Convert int16 -> float32 normalized (same as dataset)
            wav_crop_float = crop_int16.float() / 32767.0
            
            # RMS filtering (if specified) - calculate on float32 normalized waveform
            if min_rms is not None:
                rms = calculate_rms(wav_crop_float)
                if rms < min_rms:
                    continue
            
            # Convert to mel spectrogram (same as dataset's get_mel_spec)
            spec = transform(wav_crop_float)
            spec = amplitude_to_db(spec)
            chunks.append(spec)
        
        return chunks
    
    logger.info(f"\n🧪 Starting two-file test:")
    logger.info(f"   Normal file: {os.path.basename(normal_wav_path)}")
    logger.info(f"   Anomaly file: {os.path.basename(anomaly_wav_path)}")
    logger.info(f"   Calibration from first half: {half_len / sample_rate:.2f} seconds")
    logger.info(f"   Testing from second half: {(total_len - half_len) / sample_rate:.2f} seconds")
    
    # Calibration from first half
    logger.info(f"\n📊 Calibration from first half...")
    calibration_chunks_list = extract_random_chunks(
        first_half_int16,
        calibration_chunks,
        min_rms=rms_threshold
    )
    
    if len(calibration_chunks_list) < 10:
        logger.error(f"Insufficient chunks for calibration ({len(calibration_chunks_list)})")
        return None
    
    # Get embedding vectors for calibration
    calib_batch = torch.stack(calibration_chunks_list)  # [calibration_chunks, 1, 64, 32]
    calib_vecs = get_embedding(model, calib_batch, device)
    
    # Find "golden" vector and threshold
    golden_vector = torch.mean(calib_vecs, dim=0, keepdim=True)
    dists = torch.norm(calib_vecs - golden_vector, dim=1).cpu().numpy()
    mean_dist = np.mean(dists)
    std_dist = np.std(dists)
    threshold = mean_dist + 3 * std_dist
    
    logger.info(f"   Calibration: Mean={mean_dist:.4f}, Std={std_dist:.4f}, Threshold={threshold:.4f}")
    
    # Testing second half of normal file
    logger.info(f"\n🔍 Testing second half of normal file...")
    test_normal_chunks = extract_random_chunks(
        second_half_int16,
        test_chunks,
        min_rms=rms_threshold
    )
    
    normal_distances = []
    for chunk in test_normal_chunks:
        vec = get_embedding(model, chunk.unsqueeze(0), device)
        dist = torch.norm(vec - golden_vector).item()
        normal_distances.append(dist)
    
    # Testing anomaly file
    logger.info(f"🔍 Testing anomaly file...")
    test_anomaly_chunks = extract_random_chunks(
        anomaly_wav_int16,
        test_chunks,
        min_rms=rms_threshold
    )
    
    anomaly_distances = []
    for chunk in test_anomaly_chunks:
        vec = get_embedding(model, chunk.unsqueeze(0), device)
        dist = torch.norm(vec - golden_vector).item()
        anomaly_distances.append(dist)
    
    # Calculate statistics
    if len(normal_distances) == 0 or len(anomaly_distances) == 0:
        logger.error("Insufficient test data")
        return None
    
    # False Positives: normal samples classified as anomalies (dist > threshold)
    false_positives = sum(d > threshold for d in normal_distances)
    false_positive_rate = false_positives / len(normal_distances) if len(normal_distances) > 0 else 0.0
    
    # False Negatives: anomaly samples classified as normal (dist <= threshold)
    false_negatives = sum(d <= threshold for d in anomaly_distances)
    false_negative_rate = false_negatives / len(anomaly_distances) if len(anomaly_distances) > 0 else 0.0
    
    avg_dist_normal = np.mean(normal_distances)
    avg_dist_anomaly = np.mean(anomaly_distances)
    
    logger.info(f"\n📊 Test Results:")
    logger.info(f"   Normal samples tested: {len(normal_distances)}")
    logger.info(f"   Anomaly samples tested: {len(anomaly_distances)}")
    logger.info(f"   AVG Distance Normal:  {avg_dist_normal:.4f}")
    logger.info(f"   AVG Distance Anomaly: {avg_dist_anomaly:.4f}")
    logger.info(f"   Threshold: {threshold:.4f}")
    logger.info(f"   ⚠️ False Positives: {false_positives}/{len(normal_distances)} ({false_positive_rate*100:.2f}%)")
    logger.info(f"   ⚠️ False Negatives: {false_negatives}/{len(anomaly_distances)} ({false_negative_rate*100:.2f}%)")
    logger.info(f"   Normal Accuracy: {(1 - false_positive_rate)*100:.2f}%")
    logger.info(f"   Anomaly Accuracy: {(1 - false_negative_rate)*100:.2f}%")
    
    # Visualization
    if show_plot and MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 6))
        plt.hist(normal_distances, bins=30, alpha=0.7, color='green', label='Normal', density=True)
        plt.hist(anomaly_distances, bins=30, alpha=0.7, color='red', label='Anomaly', density=True)
        plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
        plt.xlabel('Distance to Golden Vector')
        plt.ylabel('Density')
        plt.title('Two-File Test Results\n(Calibration from first half of normal file)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return {
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_normal': len(normal_distances),
        'total_anomaly': len(anomaly_distances),
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'threshold': threshold,
        'mean_dist_normal': mean_dist,
        'std_dist_normal': std_dist,
        'avg_dist_normal': avg_dist_normal,
        'avg_dist_anomaly': avg_dist_anomaly,
        'normal_distances': normal_distances,
        'anomaly_distances': anomaly_distances,
        'normal_file': os.path.basename(normal_wav_path),
        'anomaly_file': os.path.basename(anomaly_wav_path)
    }


def test_module_infrastructure():
    """
    Quick test of module infrastructure to verify everything works.
    Tests MLflow connection, dataset loading, and basic functionality.
    """
    logger.info("🧪 Running module infrastructure test...")
    
    try:
        # Test 1: Check MLflow availability
        if MLFLOW_AVAILABLE:
            logger.info("✅ MLflow is available")
            try:
                mlflow.set_tracking_uri(DEFAULT_MLFLOW_TRACKING_URI)
                logger.info(f"✅ MLflow tracking URI configured: {DEFAULT_MLFLOW_TRACKING_URI}")
            except Exception as e:
                logger.warning(f"⚠️ Could not configure MLflow: {e}")
        else:
            logger.warning("⚠️ MLflow is not available")
        
        # Test 2: Check MinIO/S3 configuration
        if MLFLOW_S3_ENDPOINT_URL:
            logger.info(f"✅ MinIO/S3 endpoint configured: {MLFLOW_S3_ENDPOINT_URL}")
        else:
            logger.info("ℹ️ MinIO/S3 endpoint not configured (using default MLflow storage)")
        
        # Test 3: Check PyTorch
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        if torch.backends.mps.is_available():
            logger.info("✅ MPS (Apple Silicon) available")
        elif torch.cuda.is_available():
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("ℹ️ Using CPU")
        
        # Test 4: Check model class
        try:
            model = TinyAudioCNN(embedding_dim=64)
            logger.info("✅ TinyAudioCNN model can be instantiated")
            params = model.count_parameters()
            logger.info(f"   Model has {params['total']:,} parameters")
        except Exception as e:
            logger.error(f"❌ Failed to instantiate model: {e}")
            return False
        
        # Test 5: Check dataset class
        try:
            # Try to load a dummy dataset (will fail if file doesn't exist, but class should work)
            logger.info("✅ TripletMemoryDataset class is available")
        except Exception as e:
            logger.error(f"❌ Dataset class error: {e}")
            return False
        
        # Test 6: Check helper functions
        try:
            dummy_tensor = torch.randn(1, 16000)
            rms = calculate_rms(dummy_tensor)
            logger.info(f"✅ calculate_rms() works: RMS = {rms:.6f}")
        except Exception as e:
            logger.error(f"❌ calculate_rms() failed: {e}")
            return False
        
        logger.info("✅ All infrastructure tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Infrastructure test failed: {e}", exc_info=True)
        return False


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Test and validate TinyAudioCNN model with MLflow integration'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run quick infrastructure test (verify module setup)'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Path to model (local path, MLflow URI, or MLflow run_id) - required for validation'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=DEFAULT_DATASET_FILE,
        help=f'Path to dataset .pt file (default: {DEFAULT_DATASET_FILE})'
    )
    parser.add_argument(
        '--snr',
        type=str,
        default='6db',
        choices=['6db', '0db', '_6db'],
        help='SNR level to test (default: 6db)'
    )
    parser.add_argument(
        '--skip-types',
        type=str,
        nargs='+',
        default=[],
        help='Machine types to skip (e.g., --skip-types fan pump)'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=50,
        help='Number of validation iterations per machine type (default: 50)'
    )
    parser.add_argument(
        '--rms-threshold',
        type=float,
        default=DEFAULT_RMS_THRESHOLD,
        help=f'RMS threshold for silence filtering (default: {DEFAULT_RMS_THRESHOLD})'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f'Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})'
    )
    parser.add_argument(
        '--mlflow-run-id',
        type=str,
        default=None,
        help='MLflow run ID for logging results (default: extract from model path)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'mps'],
        default=None,
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--normal', '-n',
        type=str,
        default=None,
        help='Path to normal (good) WAV file for two-file testing mode'
    )
    parser.add_argument(
        '--anomaly', '-a',
        type=str,
        default=None,
        help='Path to anomaly (bad) WAV file for two-file testing mode'
    )
    parser.add_argument(
        '--calibration-chunks',
        type=int,
        default=50,
        help='Number of chunks for calibration (default: 50)'
    )
    parser.add_argument(
        '--test-chunks',
        type=int,
        default=50,
        help='Number of chunks for testing (default: 50)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Sample rate for WAV files (default: 16000)'
    )
    parser.add_argument(
        '--duration-sec',
        type=float,
        default=1.0,
        help='Duration of chunks in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Do not show visualization plot'
    )
    
    args = parser.parse_args()
    
    # Handle infrastructure test mode
    if args.test:
        success = test_module_infrastructure()
        return 0 if success else 1
    
    # Handle two-file testing mode
    if args.normal or args.anomaly:
        if not args.model:
            parser.print_help()
            logger.error("--model is required for two-file testing mode")
            return 1
        if not args.normal:
            parser.print_help()
            logger.error("--normal is required for two-file testing mode")
            return 1
        if not args.anomaly:
            parser.print_help()
            logger.error("--anomaly is required for two-file testing mode")
            return 1
        
        try:
            # Determine device
            if args.device:
                device = torch.device(args.device)
            else:
                if torch.backends.mps.is_available():
                    device = torch.device("mps")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                else:
                    device = torch.device("cpu")
            
            logger.info(f"Using device: {device}")
            
            # Run two-file test
            results = test_two_wav_files(
                model_path=args.model,
                normal_wav_path=args.normal,
                anomaly_wav_path=args.anomaly,
                device=device,
                sample_rate=args.sample_rate,
                duration_sec=args.duration_sec,
                calibration_chunks=args.calibration_chunks,
                test_chunks=args.test_chunks,
                rms_threshold=args.rms_threshold,
                embedding_dim=args.embedding_dim,
                show_plot=not args.no_plot
            )
            
            if results:
                logger.info("✅ Two-file test completed successfully!")
                return 0
            else:
                logger.error("❌ Two-file test failed!")
                return 1
                
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Error during two-file test: {e}", exc_info=True)
            return 1
    
    # Normal validation mode - requires model path
    if not args.model:
        parser.print_help()
        logger.error("--model is required for validation. Use --test for infrastructure test or --normal/--anomaly for two-file testing.")
        return 1
    
    try:
        # Determine device
        if args.device:
            device = torch.device(args.device)
        else:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        logger.info(f"Using device: {device}")
        
        # Run full validation
        results = run_full_validation(
            model_path=args.model,
            dataset_file=args.dataset,
            device=device,
            snr=args.snr,
            skip_types=args.skip_types,
            num_iterations=args.iterations,
            rms_threshold=args.rms_threshold,
            embedding_dim=args.embedding_dim,
            mlflow_run_id=args.mlflow_run_id
        )
        
        logger.info("✅ Validation completed successfully!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())