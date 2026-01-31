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
    from validate import validate_model, run_full_validation
    results = validate_model(model, dataset, device, 'fan', rms_threshold=0.005)

Usage as script:
    # Run validation (normal mode)
    python validate.py --model runs:/<run_id>/models/model.pt --dataset dataset.pt
    
    # Quick infrastructure test (verify module setup)
    python validate.py --test
    
    # Two-file WAV testing mode
    python validate.py --model runs:/<run_id>/models/model.pt --normal normal.wav --anomaly anomaly.wav
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
from datetime import datetime
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
    from ml.model import TinyAudioCNN, TinyAudioCNN_v2, TinyAudioCNN_v3, TinyAudioCNN_v4
    from ml.triplet_memory_dataset import TripletMemoryDataset
except ImportError:
    # Fallback to relative imports (when running as script from ml/ directory)
    from model import TinyAudioCNN, TinyAudioCNN_v2, TinyAudioCNN_v3, TinyAudioCNN_v4
    from triplet_memory_dataset import TripletMemoryDataset

# Constants (can be overridden via environment variables)
DEFAULT_DATASET_FILE = os.getenv('DEFAULT_DATASET_FILE', None)
if DEFAULT_DATASET_FILE == '': DEFAULT_DATASET_FILE = None
DEFAULT_EMBEDDING_DIM = int(os.getenv('MODEL_EMBEDDING_DIM', '64'))
DEFAULT_RMS_THRESHOLD = float(os.getenv('TEST_RMS_THRESHOLD', '0.0001'))
DEFAULT_VALIDATION_ITERATIONS = int(os.getenv('VALIDATION_ITERATIONS', '50'))
DEFAULT_ABNORMAL_RATIO = float(os.getenv('TRIPLET_ABNORMAL_RATIO', '0.5'))
DEFAULT_CALIBRATION_SAMPLES = int(os.getenv('VALIDATION_CALIBRATION_SAMPLES', '50'))
DEFAULT_THRESHOLD_STD_MULTIPLIER = float(os.getenv('VALIDATION_THRESHOLD_STD_MULTIPLIER', '3.0'))
DEFAULT_MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
DEFAULT_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'Model_testing_validation')

# Default SNR values for MIMII dataset
DEFAULT_SNR_VALUES = ['6db', '0db', '_6db']

# Try to import sklearn (required for PCA detector)
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available. PCA detector will be disabled. Install with: pip install scikit-learn")


# ============================================================================
# ANOMALY DETECTOR CLASSES
# ============================================================================

class AnomalyDetector:
    """
    Base class for anomaly detection algorithms.
    Each detector implements calibrate() and is_anomaly() methods.
    """
    
    def __init__(self, threshold_std_multiplier: float = 3.0):
        """
        Initialize detector.
        
        Args:
            threshold_std_multiplier: Multiplier for standard deviation when calculating threshold
        """
        self.threshold_std_multiplier = threshold_std_multiplier
        self.is_calibrated = False
    
    def calibrate(self, calibration_vectors: torch.Tensor) -> Dict[str, Any]:
        """
        Calibrate detector on normal (calibration) vectors.
        
        Args:
            calibration_vectors: Tensor of shape [N, embedding_dim] with normal vectors
        
        Returns:
            Dictionary with calibration statistics
        """
        raise NotImplementedError("Subclasses must implement calibrate()")
    
    def is_anomaly(self, vector: torch.Tensor) -> tuple[bool, float]:
        """
        Check if a vector is an anomaly.
        
        Args:
            vector: Tensor of shape [1, embedding_dim] or [embedding_dim]
        
        Returns:
            Tuple of (is_anomaly: bool, distance_or_score: float)
        """
        raise NotImplementedError("Subclasses must implement is_anomaly()")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get detector information and calibration parameters.
        
        Returns:
            Dictionary with detector info
        """
        return {
            'detector_type': self.__class__.__name__,
            'is_calibrated': self.is_calibrated,
            'threshold_std_multiplier': self.threshold_std_multiplier,
        }


class MahalanobisDetector(AnomalyDetector):
    """
    Anomaly detector using weighted (Mahalanobis-style) distance.
    
    Uses standardized distance: sqrt(sum((x_i - mean_i) / std_i)^2)
    This accounts for different variances across dimensions.
    """
    
    def __init__(self, threshold_std_multiplier: float = 3.0):
        super().__init__(threshold_std_multiplier)
        self.golden_vector = None  # [1, embedding_dim]
        self.std_vector = None  # [1, embedding_dim]
        self.threshold = None
        self.mean_dist = None
        self.std_dist = None
    
    def calibrate(self, calibration_vectors: torch.Tensor) -> Dict[str, Any]:
        """
        Calibrate using weighted Mahalanobis distance.
        
        Args:
            calibration_vectors: Tensor of shape [N, embedding_dim]
        
        Returns:
            Dictionary with calibration statistics
        """
        # Calculate mean vector (centroid) and std vector (standard deviation per dimension)
        self.golden_vector = torch.mean(calibration_vectors, dim=0, keepdim=True)  # [1, embedding_dim]
        self.std_vector = torch.std(calibration_vectors, dim=0, keepdim=True)  # [1, embedding_dim]
        
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        self.std_vector = self.std_vector + epsilon
        
        # Calculate weighted (standardized) distances for calibration
        # Weighted distance: sqrt(sum((x_i - mean_i) / std_i)^2)
        standardized_diff = (calibration_vectors - self.golden_vector) / self.std_vector
        dists = torch.norm(standardized_diff, dim=1).cpu().numpy()
        
        self.mean_dist = float(np.mean(dists))
        self.std_dist = float(np.std(dists))
        self.threshold = self.mean_dist + self.threshold_std_multiplier * self.std_dist
        self.is_calibrated = True
        
        return {
            'mean_dist': self.mean_dist,
            'std_dist': self.std_dist,
            'threshold': self.threshold,
            'golden_vector_shape': list(self.golden_vector.shape),
            'std_vector_shape': list(self.std_vector.shape),
        }
    
    def is_anomaly(self, vector: torch.Tensor) -> tuple[bool, float]:
        """
        Check if vector is anomaly using weighted distance.
        
        Args:
            vector: Tensor of shape [1, embedding_dim] or [embedding_dim]
        
        Returns:
            Tuple of (is_anomaly: bool, distance: float)
        """
        if not self.is_calibrated:
            raise RuntimeError("Detector not calibrated. Call calibrate() first.")
        
        # Ensure vector has correct shape
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)  # [embedding_dim] -> [1, embedding_dim]
        
        # Calculate weighted (standardized) distance
        standardized_diff = (vector - self.golden_vector) / self.std_vector
        dist = torch.norm(standardized_diff, dim=1).item()
        
        is_anomaly = dist > self.threshold
        return is_anomaly, dist
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        if self.is_calibrated:
            info.update({
                'mean_dist': self.mean_dist,
                'std_dist': self.std_dist,
                'threshold': self.threshold,
            })
        return info


class PCADetector(AnomalyDetector):
    """
    Anomaly detector using PCA projection with bounding box.
    
    Projects high-dimensional vectors to 3D space using PCA,
    then checks if the projected point is within a bounding box
    defined by mean ± sigma_threshold * std for each principal component.
    """
    
    def __init__(self, n_components: int = 3, threshold_std_multiplier: float = 3.0):
        """
        Initialize PCA detector.
        
        Args:
            n_components: Number of principal components to use (default: 3)
            threshold_std_multiplier: Multiplier for standard deviation (default: 3.0)
        """
        super().__init__(threshold_std_multiplier)
        self.n_components = n_components
        self.pca_matrix = None  # [n_components, embedding_dim]
        self.mean_vector = None  # [embedding_dim]
        self.bounds_min = None  # [n_components]
        self.bounds_max = None  # [n_components]
        self.proj_mean = None  # [n_components]
        self.proj_std = None  # [n_components]
    
    def calibrate(self, calibration_vectors: torch.Tensor) -> Dict[str, Any]:
        """
        Calibrate using PCA projection.
        
        Args:
            calibration_vectors: Tensor of shape [N, embedding_dim]
        
        Returns:
            Dictionary with calibration statistics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn is required for PCA detector. Install with: pip install scikit-learn")
        
        # Store device to move tensors back to the same device
        device = calibration_vectors.device
        
        # Convert to numpy for sklearn
        calib_np = calibration_vectors.cpu().numpy()
        
        # Fit PCA
        pca = PCA(n_components=self.n_components)
        pca_proj = pca.fit_transform(calib_np)
        
        # Store PCA parameters (what would be stored in ESP32 RAM)
        # Move to the same device as input vectors
        self.pca_matrix = torch.from_numpy(pca.components_).float().to(device)  # [n_components, embedding_dim]
        self.mean_vector = torch.from_numpy(pca.mean_).float().to(device)  # [embedding_dim]
        
        # Calculate bounds in PCA space
        self.proj_mean = np.mean(pca_proj, axis=0)  # [n_components]
        self.proj_std = np.std(pca_proj, axis=0)  # [n_components]
        
        # Bounding box: mean ± threshold_std_multiplier * std
        self.bounds_min = self.proj_mean - (self.threshold_std_multiplier * self.proj_std)
        self.bounds_max = self.proj_mean + (self.threshold_std_multiplier * self.proj_std)
        
        self.is_calibrated = True
        
        return {
            'n_components': self.n_components,
            'pca_matrix_shape': list(self.pca_matrix.shape),
            'mean_vector_shape': list(self.mean_vector.shape),
            'bounds_min': self.bounds_min.tolist(),
            'bounds_max': self.bounds_max.tolist(),
            'proj_mean': self.proj_mean.tolist(),
            'proj_std': self.proj_std.tolist(),
        }
    
    def is_anomaly(self, vector: torch.Tensor) -> tuple[bool, float]:
        """
        Check if vector is anomaly using PCA projection.
        
        Args:
            vector: Tensor of shape [1, embedding_dim] or [embedding_dim]
        
        Returns:
            Tuple of (is_anomaly: bool, max_deviation_score: float)
            max_deviation_score is the maximum normalized deviation across all components
        """
        if not self.is_calibrated:
            raise RuntimeError("Detector not calibrated. Call calibrate() first.")
        
        # Ensure vector has correct shape
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)  # [embedding_dim] -> [1, embedding_dim]
        
        # Center vector
        centered = vector - self.mean_vector  # [1, embedding_dim]
        
        # Project to PCA space: [1, embedding_dim] @ [embedding_dim, n_components] -> [1, n_components]
        point_pca = torch.matmul(centered, self.pca_matrix.T)  # [1, n_components]
        point_pca = point_pca.squeeze(0).cpu().numpy()  # [n_components]
        
        # Check if point is within bounding box
        within_bounds = np.all(
            (self.bounds_min <= point_pca) & (point_pca <= self.bounds_max)
        )
        
        # Calculate max deviation score (for logging/debugging)
        # Normalized deviation: (point - mean) / std for each component
        normalized_deviations = np.abs((point_pca - self.proj_mean) / (self.proj_std + 1e-8))
        max_deviation = float(np.max(normalized_deviations))
        
        is_anomaly = not within_bounds
        return is_anomaly, max_deviation
    
    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        if self.is_calibrated:
            info.update({
                'n_components': self.n_components,
                'bounds_min': self.bounds_min.tolist(),
                'bounds_max': self.bounds_max.tolist(),
            })
        return info


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
    
    Automatically detects model version from MLflow tags/params if loading from MLflow.
    Falls back to v1 (TinyAudioCNN) if version cannot be determined.
    
    Args:
        model_path: Path to model. Can be:
            - Local path: "/path/to/model.pth"
            - MLflow URI: "runs:/<run_id>/models/<filename>"
            - MLflow run_id: "<run_id>" (will load best_model.pth from artifacts)
        device: Target device (torch.device)
        embedding_dim: Size of embedding vector (default: 64)
    
    Returns:
        Loaded model (TinyAudioCNN or TinyAudioCNN_v2) in eval mode
    """
    logger.info(f"Loading model from {model_path}...")
    
    model_version = None
    run_id = None
    
    # Check if this is an MLflow URI
    if model_path.startswith("runs:/"):
        # MLflow URI format: runs:/<run_id>/models/<filename>
        logger.info("Loading model from MLflow...")
        try:
            # Extract run_id from URI
            parts = model_path.split("/")
            if len(parts) >= 2:
                run_id = parts[1]
            
            # Try to get model version from MLflow run tags/params
            if MLFLOW_AVAILABLE and run_id:
                try:
                    client = mlflow.tracking.MlflowClient()
                    run_data = client.get_run(run_id)
                    
                    # Check tags first (model_version tag)
                    if 'model_version' in run_data.data.tags:
                        model_version = run_data.data.tags['model_version'].lower()
                        logger.info(f"Detected model version from MLflow tags: {model_version}")
                    # Check params (model_version param)
                    elif 'model_version' in run_data.data.params:
                        model_version = run_data.data.params['model_version'].lower()
                        logger.info(f"Detected model version from MLflow params: {model_version}")
                    # Check model_type tag (TinyAudioCNN_v2/v3/v4 indicates version 2/3/4)
                    elif 'model_type' in run_data.data.tags:
                        model_type = run_data.data.tags['model_type']
                        if 'v4' in model_type.lower() or 'TinyAudioCNN_v4' in model_type or 'MBConv' in model_type or 'SE-Attention' in model_type:
                            model_version = '4'
                            logger.info(f"Detected model version from model_type tag: {model_version}")
                        elif 'v3' in model_type.lower() or 'TinyAudioCNN_v3' in model_type or 'MobileNetV2' in model_type:
                            model_version = '3'
                            logger.info(f"Detected model version from model_type tag: {model_version}")
                        elif 'v2' in model_type.lower() or 'TinyAudioCNN_v2' in model_type or '2' in model_type:
                            model_version = '2'
                            logger.info(f"Detected model version from model_type tag: {model_version}")
                except Exception as e:
                    logger.warning(f"Could not determine model version from MLflow: {e}. Using default 1.")
            
            # Download artifact from MLflow
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_path)
            logger.info(f"Model downloaded from MLflow: {local_path}")
            model_path = local_path
        except Exception as e:
            logger.error(f"Error downloading from MLflow: {e}")
            raise
    elif len(model_path) == 32 and all(c in '0123456789abcdef' for c in model_path.lower()):
        # Possibly a run_id (32 hex characters)
        run_id = model_path
        logger.info(f"Loading model from MLflow run_id: {run_id}")
        try:
            # Try to get model version from MLflow run tags/params
            if MLFLOW_AVAILABLE:
                try:
                    client = mlflow.tracking.MlflowClient()
                    run_data = client.get_run(run_id)
                    
                    # Check tags first (model_version tag)
                    if 'model_version' in run_data.data.tags:
                        model_version = run_data.data.tags['model_version'].lower().strip()
                        # Support both old format ('v1', 'v2') and new format ('1', '2')
                        if model_version.startswith('v'):
                            model_version = model_version[1:]  # Remove 'v' prefix
                        logger.info(f"Detected model version from MLflow tags: {model_version}")
                    # Check params (model_version param)
                    elif 'model_version' in run_data.data.params:
                        model_version = run_data.data.params['model_version'].lower().strip()
                        # Support both old format ('v1', 'v2') and new format ('1', '2')
                        if model_version.startswith('v'):
                            model_version = model_version[1:]  # Remove 'v' prefix
                        logger.info(f"Detected model version from MLflow params: {model_version}")
                    # Check model_type tag (TinyAudioCNN_v2/v3/v4 indicates version 2/3/4)
                    elif 'model_type' in run_data.data.tags:
                        model_type = run_data.data.tags['model_type']
                        if 'v4' in model_type.lower() or 'TinyAudioCNN_v4' in model_type or 'MBConv' in model_type or 'SE-Attention' in model_type:
                            model_version = '4'
                            logger.info(f"Detected model version from model_type tag: {model_version}")
                        elif 'v3' in model_type.lower() or 'TinyAudioCNN_v3' in model_type or 'MobileNetV2' in model_type:
                            model_version = '3'
                            logger.info(f"Detected model version from model_type tag: {model_version}")
                        elif 'v2' in model_type.lower() or 'TinyAudioCNN_v2' in model_type or '2' in model_type:
                            model_version = '2'
                            logger.info(f"Detected model version from model_type tag: {model_version}")
                except Exception as e:
                    logger.warning(f"Could not determine model version from MLflow: {e}. Using default 1.")
            
            # Look for best_model in artifacts
            artifact_uri = f"runs:/{run_id}/models"
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
    
    # Normalize model_version (support both 'v1'/'v2'/'v3' and '1'/'2'/'3' formats)
    if model_version:
        model_version = str(model_version).lower().strip()
        if model_version.startswith('v'):
            model_version = model_version[1:]  # Remove 'v' prefix if present
    
    # Determine model class based on detected version
    # Default to version 1 if version cannot be determined
    if model_version == '4':
        ModelClass = TinyAudioCNN_v4
        model_name = 'TinyAudioCNN_v4'
    elif model_version == '3':
        ModelClass = TinyAudioCNN_v3
        model_name = 'TinyAudioCNN_v3'
    elif model_version == '2':
        ModelClass = TinyAudioCNN_v2
        model_name = 'TinyAudioCNN_v2'
    else:
        ModelClass = TinyAudioCNN
        model_name = 'TinyAudioCNN'
        if model_version is None:
            logger.info("Model version not specified, using default: 1 (TinyAudioCNN)")
    
    # Load model
    logger.info(f"Instantiating model: {model_name}")
    model = ModelClass(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"✅ Model loaded successfully! ({model_name})")
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
    silent: bool = False,
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    threshold_std_multiplier: float = DEFAULT_THRESHOLD_STD_MULTIPLIER,
    detector_type: str = 'mahalanobis',
    detector: Optional[AnomalyDetector] = None
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
        calibration_samples: Number of mel spectrograms to use for calibration (default: 50, can be set via VALIDATION_CALIBRATION_SAMPLES env var)
        threshold_std_multiplier: Multiplier for standard deviation when calculating anomaly threshold (default: 3.0, can be set via VALIDATION_THRESHOLD_STD_MULTIPLIER env var). Threshold = mean_distance + threshold_std_multiplier * std_distance
    
    Returns:
        Dictionary with validation results or None if validation failed
    """
    if fixed_target_id is None:
        target_ids = dataset.ids_by_type[target_type]
        target_id = random.choice(target_ids)
    else:
        target_id = fixed_target_id

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
        logger.info(f"🟢 Target: {target_id}")

    # --- 1. CALIBRATION WITH RMS FILTER ---
    # Calibration happens on only one sample. Extract X random chunks of 1 second each
    calibration_mels = []
    # Get base sample for calibration
    base_wav, base_wav_id = dataset.get_sample(target_type, target_id, snr, normal='normal')
    attempts = 0
    while len(calibration_mels) < calibration_samples and attempts < 500:
        attempts += 1
        w = dataset.get_random_crop(base_wav)
        
        # RMS check
        rms = calculate_rms(w)
        if rms < rms_threshold:
            continue

        s = dataset.get_mel_spec(w)
        calibration_mels.append(s)
    
    if len(calibration_mels) < calibration_samples:
        logger.warning("Failed to find enough loud samples for calibration!")
        return None

    calib_batch = torch.stack(calibration_mels)
    calib_vecs = get_embedding(model, calib_batch, device)
    
    # Initialize detector if not provided
    if detector is None:
        if detector_type.lower() == 'pca':
            if not SKLEARN_AVAILABLE:
                logger.warning("sklearn not available, falling back to mahalanobis detector")
                detector = MahalanobisDetector(threshold_std_multiplier=threshold_std_multiplier)
            else:
                detector = PCADetector(n_components=3, threshold_std_multiplier=threshold_std_multiplier)
        else:
            detector = MahalanobisDetector(threshold_std_multiplier=threshold_std_multiplier)
    
    # Calibrate detector
    calib_stats = detector.calibrate(calib_vecs)
    
    if not silent:
        detector_info = detector.get_info()
        logger.info(f"📊 Calibration (RMS > {rms_threshold}) - {target_type}-{target_id}-{snr}-normal[{base_wav_id}]:")
        logger.info(f"   Detector: {detector_info['detector_type']}")
        if 'mean_dist' in calib_stats:
            logger.info(f"   Mean: {calib_stats['mean_dist']:.4f} | Std: {calib_stats['std_dist']:.4f} | Threshold: {calib_stats['threshold']:.4f}")
        elif 'bounds_min' in calib_stats:
            logger.info(f"   PCA bounds: PC1[{calib_stats['bounds_min'][0]:.4f}, {calib_stats['bounds_max'][0]:.4f}], "
                       f"PC2[{calib_stats['bounds_min'][1]:.4f}, {calib_stats['bounds_max'][1]:.4f}], "
                       f"PC3[{calib_stats['bounds_min'][2]:.4f}, {calib_stats['bounds_max'][2]:.4f}]")

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
            v = get_embedding(model, s.unsqueeze(0), device)  # [1, embedding_dim]
            
            # Use detector to check if anomaly and get distance/score
            is_anom, dist_or_score = detector.is_anomaly(v)
            result_list.append(dist_or_score)
            count += 1
        return count

    n_norm = run_test_loop(target_id, 'normal', base_wav_id, results['normal'])
    n_anom = run_test_loop(target_id, 'abnormal', None, results['anomaly'])
    
    if not silent:
        logger.info(f"🔍 Test (found loud samples): Normal={n_norm}, Anomaly={n_anom}")

    # Statistics
    validation_results = None
    if len(results['normal']) > 0 and len(results['anomaly']) > 0:
        # Re-check samples using detector (for accurate statistics)
        false_positives = 0
        false_negatives = 0
        
        # Re-check normal samples
        for dist_or_score in results['normal']:
            # For MahalanobisDetector, we have distance values
            # For PCADetector, we have max_deviation scores
            if isinstance(detector, MahalanobisDetector):
                is_anom = dist_or_score > detector.threshold
            else:  # PCADetector
                # We stored max_deviation, check if it exceeds threshold_std_multiplier
                is_anom = dist_or_score > detector.threshold_std_multiplier
            if is_anom:
                false_positives += 1
        
        # Re-check anomaly samples
        for dist_or_score in results['anomaly']:
            if isinstance(detector, MahalanobisDetector):
                is_anom = dist_or_score > detector.threshold
            else:  # PCADetector
                is_anom = dist_or_score > detector.threshold_std_multiplier
            if not is_anom:
                false_negatives += 1

        avg_dist_normal = np.mean(results['normal'])
        avg_dist_anomaly = np.mean(results['anomaly'])
        
        if not silent:
            logger.info(f"   AVG Dist Normal:  {avg_dist_normal:.4f}")
            logger.info(f"   AVG Dist Anomaly: {avg_dist_anomaly:.4f}")
            logger.info(f"   ⚠️ False Positives: {false_positives}/{n_norm}")
            logger.info(f"   ⚠️ Missed Anomalies: {false_negatives}/{n_anom}")

        # Prepare results for MLflow
        detector_info = detector.get_info()
        validation_results = {
            'target_type': target_type,
            'snr': snr,
            'target_id': target_id,
            'detector_type': detector_info['detector_type'],
            'avg_dist_normal': float(avg_dist_normal),
            'avg_dist_anomaly': float(avg_dist_anomaly),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives),
            'total_normal': int(n_norm),
            'total_anomaly': int(n_anom),
            'false_positive_rate': float(false_positives / n_norm) if n_norm > 0 else 0.0,
            'false_negative_rate': float(false_negatives / n_anom) if n_anom > 0 else 0.0,
        }
        
        # Add detector-specific metrics
        detector_info = detector.get_info()
        if isinstance(detector, MahalanobisDetector):
            validation_results.update({
                'threshold': float(detector.threshold),
                'mean_dist_normal': float(detector.mean_dist),
                'std_dist_normal': float(detector.std_dist),
            })
        elif isinstance(detector, PCADetector):
            validation_results.update({
                'n_components': detector.n_components,
                'threshold_std_multiplier': detector.threshold_std_multiplier,
            })

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
                metrics = {
                    f'validation_{target_type}_{snr}_avg_dist_normal': avg_dist_normal,
                    f'validation_{target_type}_{snr}_avg_dist_anomaly': avg_dist_anomaly,
                    f'validation_{target_type}_{snr}_false_positive_rate': validation_results['false_positive_rate'],
                    f'validation_{target_type}_{snr}_false_negative_rate': validation_results['false_negative_rate'],
                    f'validation_{target_type}_{snr}_false_positives': false_positives,
                    f'validation_{target_type}_{snr}_false_negatives': false_negatives,
                }
                # Add threshold only for MahalanobisDetector
                if isinstance(detector, MahalanobisDetector):
                    metrics[f'validation_{target_type}_{snr}_threshold'] = detector.threshold
                mlflow.log_metrics(metrics)
                
                # Log validation parameters
                mlflow.log_params({
                    f'validation_{target_type}_{snr}_target_id': target_id,
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
                    # Draw threshold line based on detector type
                    if isinstance(detector, MahalanobisDetector):
                        plt.axvline(detector.threshold, color='black', linestyle='--', label=f'Threshold ({detector.threshold:.4f})')
                    plt.title(f"{target_type} ({snr}) - {detector_info['detector_type']} - RMS Filtered")
                    plt.xlabel('Distance/Score')
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
            # Draw threshold line based on detector type
            if isinstance(detector, MahalanobisDetector):
                plt.axvline(detector.threshold, color='black', linestyle='--', label=f'Threshold ({detector.threshold:.4f})')
            plt.title(f"{target_type} - {detector_info['detector_type']}")
            plt.xlabel('Distance/Score')
            plt.legend()
            plt.show()
    else:
        logger.warning("Insufficient data after filtering.")
    
    return validation_results


def run_full_validation(
    model_path: str,
    dataset_file: Optional[str] = DEFAULT_DATASET_FILE,
    device: Optional[torch.device] = None,
    snr: str = '0db',
    skip_types: List[str] = None,
    num_iterations: int = DEFAULT_VALIDATION_ITERATIONS,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    mlflow_run_id: Optional[str] = None,
    abnormal_ratio: float = DEFAULT_ABNORMAL_RATIO,
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    threshold_std_multiplier: float = DEFAULT_THRESHOLD_STD_MULTIPLIER
) -> Dict[str, float]:
    """
    Run full validation across all machine types with multiple iterations.
    
    Args:
        model_path: Path to model (local or MLflow URI)
        dataset_file: Path to dataset .pt file. If not provided, uses DEFAULT_DATASET_FILE from environment variable.
            If DEFAULT_DATASET_FILE is not set, raises ValueError.
        device: Device to use (None = auto-detect)
        snr: SNR level to test ('6db', '0db', '_6db')
        skip_types: List of machine types to skip
        num_iterations: Number of validation iterations per machine type (default: 50, can be set via VALIDATION_ITERATIONS env var)
        rms_threshold: RMS threshold for silence filtering
        embedding_dim: Embedding dimension of the model
        mlflow_run_id: MLflow run ID for logging results
        abnormal_ratio: Ratio of abnormal samples to use as negatives (default: 0.5, i.e., 50% abnormal negatives, 50% different machine, can be set via TRIPLET_ABNORMAL_RATIO env var)
        calibration_samples: Number of mel spectrograms to use for calibration (default: 50, can be set via VALIDATION_CALIBRATION_SAMPLES env var)
        threshold_std_multiplier: Multiplier for standard deviation when calculating anomaly threshold (default: 3.0, can be set via VALIDATION_THRESHOLD_STD_MULTIPLIER env var). Threshold = mean_distance + threshold_std_multiplier * std_distance
    
    Returns:
        Dictionary with average false_positive_rate and false_negative_rate
    
    Raises:
        ValueError: If dataset_file is not provided and DEFAULT_DATASET_FILE is not set.
    """
    # Validate dataset_file
    if dataset_file is None:
        raise ValueError(
            "dataset_file is required. Please provide it as an argument or set "
            "DEFAULT_DATASET_FILE environment variable."
        )
    
    # Setup file logging for validation
    log_file_handler = None
    log_file_path = None
    
    try:
        # Extract dataset name from path for log file naming
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        # Replace spaces and special characters with underscores for filename safety
        dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # Create temporary log file with dataset name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = tempfile.NamedTemporaryFile(
            mode='w',
            prefix=f'{dataset_name}_validation_',
            suffix=f'_{timestamp}.log',
            delete=False
        ).name
        
        # Create file handler with same format as console handler
        log_file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        log_file_handler.setLevel(logging.DEBUG)  # Log everything to file
        log_file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        
        # Add file handler to logger
        logger.addHandler(log_file_handler)
        logger.info(f"Validation log file: {log_file_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create log file handler: {e}")
        log_file_handler = None
    
    try:
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
        logger.info(f"Using abnormal_ratio: {abnormal_ratio}")
        dataset = TripletMemoryDataset(dataset_file, samples_per_epoch=100, skip_types=skip_types, abnormal_ratio=abnormal_ratio)
        
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
                    mlflow_run_id=current_mlflow_run_id, silent=silent,
                    calibration_samples=calibration_samples,
                    threshold_std_multiplier=threshold_std_multiplier
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

                logger.info(f"Total validation iterations: {count}")
                logger.info(f"Total False Positive Rate: {total_results['false_positive_rate']:.4f}")
                logger.info(f"Total False Negative Rate: {total_results['false_negative_rate']:.4f}")

                # Log metrics
                mlflow.log_metrics({
                    'total_false_positive_rate': total_results['false_positive_rate'],
                    'total_false_negative_rate': total_results['false_negative_rate'],
                })
                
                # Save validation log file as artifact
                if log_file_path and os.path.exists(log_file_path):
                    try:
                        mlflow.log_artifact(log_file_path, "logs")
                        logger.info(f"Validation log file saved to MLflow artifacts: logs/{os.path.basename(log_file_path)}")
                    except Exception as e:
                        logger.warning(f"Failed to save log file to MLflow: {e}")
                
                if started_run:
                    mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to log total results to MLflow: {e}")
        else:
            logger.info(f"Total False Positive Rate: {total_results['false_positive_rate']:.4f}")
            logger.info(f"Total False Negative Rate: {total_results['false_negative_rate']:.4f}")
        
        return total_results
    
    finally:
        # Clean up file handler
        if log_file_handler:
            try:
                logger.removeHandler(log_file_handler)
                log_file_handler.close()
            except Exception as e:
                logger.warning(f"Failed to remove log file handler: {e}")
        
        # Clean up log file (it's already saved to MLflow)
        if log_file_path and os.path.exists(log_file_path):
            try:
                os.unlink(log_file_path)
                logger.debug(f"Removed temporary log file: {log_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary log file: {e}")


def test_two_wav_files(
    model_path: str,
    normal_wav_path: str,
    anomaly_wav_path: str,
    device: Optional[torch.device] = None,
    sample_rate: int = 16000,
    duration_sec: float = 1.0,
    calibration_samples: int = DEFAULT_CALIBRATION_SAMPLES,
    threshold_std_multiplier: float = DEFAULT_THRESHOLD_STD_MULTIPLIER,
    test_chunks: int = 50,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    show_plot: bool = True,
    detector_type: str = 'mahalanobis',
    detector: Optional[AnomalyDetector] = None
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
        calibration_samples: Number of chunks for calibration (default: 50)
        threshold_std_multiplier: Multiplier for standard deviation when calculating anomaly threshold (default: 3.0, can be set via VALIDATION_THRESHOLD_STD_MULTIPLIER env var). Threshold = mean_distance + threshold_std_multiplier * std_distance
        test_chunks: Number of chunks for testing (default: 50)
        rms_threshold: RMS threshold for silence filtering (default: 0.005)
        embedding_dim: Embedding dimension (default: 64)
        show_plot: Whether to show visualization plot (default: True)
        detector_type: Type of detector to use ('mahalanobis' or 'pca'). Ignored if detector is provided.
        detector: Optional pre-configured detector instance. If None, a new detector will be created based on detector_type.
    
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
    transform, amplitude_to_log = TripletMemoryDataset.create_mel_transform(sample_rate)
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
            spec = amplitude_to_log(spec)
            chunks.append(spec)
        
        return chunks
    
    logger.info(f"\n🧪 Starting two-file test:")
    logger.info(f"   Normal file: {os.path.basename(normal_wav_path)}")
    logger.info(f"   Anomaly file: {os.path.basename(anomaly_wav_path)}")
    logger.info(f"   Calibration from first half: {half_len / sample_rate:.2f} seconds")
    logger.info(f"   Testing from second half: {(total_len - half_len) / sample_rate:.2f} seconds")
    
    # Calibration from first half
    logger.info(f"\n📊 Calibration from first half...")
    calibration_samples_list = extract_random_chunks(
        first_half_int16,
        calibration_samples,
        min_rms=rms_threshold
    )
    
    if len(calibration_samples_list) < 10:
        logger.error(f"Insufficient chunks for calibration ({len(calibration_samples_list)})")
        return None
    
    # Get embedding vectors for calibration
    calib_batch = torch.stack(calibration_samples_list)  # [calibration_samples, 1, 64, 32]
    calib_vecs = get_embedding(model, calib_batch, device)
    
    # Initialize detector if not provided
    if detector is None:
        if detector_type.lower() == 'pca':
            if not SKLEARN_AVAILABLE:
                logger.warning("sklearn not available, falling back to mahalanobis detector")
                detector = MahalanobisDetector(threshold_std_multiplier=threshold_std_multiplier)
            else:
                detector = PCADetector(n_components=3, threshold_std_multiplier=threshold_std_multiplier)
        else:
            detector = MahalanobisDetector(threshold_std_multiplier=threshold_std_multiplier)
    
    # Calibrate detector
    calib_stats = detector.calibrate(calib_vecs)
    detector_info = detector.get_info()
    
    logger.info(f"   Detector: {detector_info['detector_type']}")
    if 'mean_dist' in calib_stats:
        logger.info(f"   Calibration: Mean={calib_stats['mean_dist']:.4f}, Std={calib_stats['std_dist']:.4f}, Threshold={calib_stats['threshold']:.4f}")
    elif 'bounds_min' in calib_stats:
        logger.info(f"   PCA bounds: PC1[{calib_stats['bounds_min'][0]:.4f}, {calib_stats['bounds_max'][0]:.4f}], "
                   f"PC2[{calib_stats['bounds_min'][1]:.4f}, {calib_stats['bounds_max'][1]:.4f}], "
                   f"PC3[{calib_stats['bounds_min'][2]:.4f}, {calib_stats['bounds_max'][2]:.4f}]")
    
    # Testing second half of normal file
    logger.info(f"\n🔍 Testing second half of normal file...")
    test_normal_chunks = extract_random_chunks(
        second_half_int16,
        test_chunks,
        min_rms=rms_threshold
    )
    
    normal_distances = []
    for chunk in test_normal_chunks:
        vec = get_embedding(model, chunk.unsqueeze(0), device)  # [1, embedding_dim]
        
        # Use detector to check if anomaly and get distance/score
        is_anom, dist_or_score = detector.is_anomaly(vec)
        normal_distances.append(dist_or_score)
    
    # Testing anomaly file
    logger.info(f"🔍 Testing anomaly file...")
    test_anomaly_chunks = extract_random_chunks(
        anomaly_wav_int16,
        test_chunks,
        min_rms=rms_threshold
    )
    
    anomaly_distances = []
    for chunk in test_anomaly_chunks:
        vec = get_embedding(model, chunk.unsqueeze(0), device)  # [1, embedding_dim]
        
        # Use detector to check if anomaly and get distance/score
        is_anom, dist_or_score = detector.is_anomaly(vec)
        anomaly_distances.append(dist_or_score)
    
    # Calculate statistics
    if len(normal_distances) == 0 or len(anomaly_distances) == 0:
        logger.error("Insufficient test data")
        return None
    
    # Calculate false positives and false negatives using detector
    false_positives = 0
    false_negatives = 0
    
    # Re-check normal samples
    for dist_or_score in normal_distances:
        if isinstance(detector, MahalanobisDetector):
            is_anom = dist_or_score > detector.threshold
        else:  # PCADetector
            is_anom = dist_or_score > detector.threshold_std_multiplier
        if is_anom:
            false_positives += 1
    
    # Re-check anomaly samples
    for dist_or_score in anomaly_distances:
        if isinstance(detector, MahalanobisDetector):
            is_anom = dist_or_score > detector.threshold
        else:  # PCADetector
            is_anom = dist_or_score > detector.threshold_std_multiplier
        if not is_anom:
            false_negatives += 1
    
    false_positive_rate = false_positives / len(normal_distances) if len(normal_distances) > 0 else 0.0
    false_negative_rate = false_negatives / len(anomaly_distances) if len(anomaly_distances) > 0 else 0.0
    
    avg_dist_normal = np.mean(normal_distances)
    avg_dist_anomaly = np.mean(anomaly_distances)
    
    logger.info(f"\n📊 Test Results:")
    logger.info(f"   Normal samples tested: {len(normal_distances)}")
    logger.info(f"   Anomaly samples tested: {len(anomaly_distances)}")
    logger.info(f"   AVG Distance Normal:  {avg_dist_normal:.4f}")
    logger.info(f"   AVG Distance Anomaly: {avg_dist_anomaly:.4f}")
    # Log threshold info based on detector type
    if isinstance(detector, MahalanobisDetector):
        logger.info(f"   Threshold: {detector.threshold:.4f}")
    elif isinstance(detector, PCADetector):
        logger.info(f"   Threshold multiplier: {detector.threshold_std_multiplier}")
    logger.info(f"   ⚠️ False Positives: {false_positives}/{len(normal_distances)} ({false_positive_rate*100:.2f}%)")
    logger.info(f"   ⚠️ False Negatives: {false_negatives}/{len(anomaly_distances)} ({false_negative_rate*100:.2f}%)")
    logger.info(f"   Normal Accuracy: {(1 - false_positive_rate)*100:.2f}%")
    logger.info(f"   Anomaly Accuracy: {(1 - false_negative_rate)*100:.2f}%")
    
    # Visualization
    if show_plot and MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(12, 6))
        plt.hist(normal_distances, bins=30, alpha=0.7, color='green', label='Normal', density=True)
        plt.hist(anomaly_distances, bins=30, alpha=0.7, color='red', label='Anomaly', density=True)
        # Draw threshold line based on detector type
        if isinstance(detector, MahalanobisDetector):
            plt.axvline(detector.threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({detector.threshold:.4f})')
        plt.xlabel('Distance/Score')
        plt.ylabel('Density')
        plt.title(f'Two-File Test Results - {detector_info["detector_type"]}\n(Calibration from first half of normal file)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    result = {
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_normal': len(normal_distances),
        'total_anomaly': len(anomaly_distances),
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'detector_type': detector_info['detector_type'],
        'avg_dist_normal': avg_dist_normal,
        'avg_dist_anomaly': avg_dist_anomaly,
        'normal_distances': normal_distances,
        'anomaly_distances': anomaly_distances,
        'normal_file': os.path.basename(normal_wav_path),
        'anomaly_file': os.path.basename(anomaly_wav_path)
    }
    
    # Add detector-specific metrics
    if isinstance(detector, MahalanobisDetector):
        result.update({
            'threshold': detector.threshold,
            'mean_dist_normal': detector.mean_dist,
            'std_dist_normal': detector.std_dist,
        })
    elif isinstance(detector, PCADetector):
        result.update({
            'n_components': detector.n_components,
            'threshold_std_multiplier': detector.threshold_std_multiplier,
        })
    
    return result


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
        
        # Test 4: Check model classes
        try:
            model_v1 = TinyAudioCNN(embedding_dim=64)
            logger.info("✅ TinyAudioCNN (v1) model can be instantiated")
            params_v1 = model_v1.count_parameters()
            logger.info(f"   V1 model has {params_v1['total']:,} parameters")
            
            model_v2 = TinyAudioCNN_v2(embedding_dim=64)
            logger.info("✅ TinyAudioCNN_v2 (v2) model can be instantiated")
            params_v2 = model_v2.count_parameters()
            logger.info(f"   V2 model has {params_v2['total']:,} parameters")
            logger.info(f"   V2 reduction: {params_v1['total'] / params_v2['total']:.2f}x")
            
            model_v3 = TinyAudioCNN_v3(embedding_dim=64)
            logger.info("✅ TinyAudioCNN_v3 (v3) model can be instantiated")
            params_v3 = model_v3.count_parameters()
            logger.info(f"   V3 model has {params_v3['total']:,} parameters")
            logger.info(f"   V3 vs V1: {params_v3['total'] / params_v1['total']:.2f}x")
            logger.info(f"   V3 vs V2: {params_v3['total'] / params_v2['total']:.2f}x")
            
            model_v4 = TinyAudioCNN_v4(embedding_dim=64)
            logger.info("✅ TinyAudioCNN_v4 (v4) model can be instantiated")
            params_v4 = model_v4.count_parameters()
            logger.info(f"   V4 model has {params_v4['total']:,} parameters")
            logger.info(f"   V4 vs V1: {params_v4['total'] / params_v1['total']:.2f}x")
            logger.info(f"   V4 vs V3: {params_v4['total'] / params_v3['total']:.2f}x")
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
        default=DEFAULT_VALIDATION_ITERATIONS,
        help=f'Number of validation iterations per machine type (default: {DEFAULT_VALIDATION_ITERATIONS}, can be set via VALIDATION_ITERATIONS env var)'
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
        '--abnormal-ratio',
        type=float,
        default=DEFAULT_ABNORMAL_RATIO,
        help=f'Ratio of abnormal samples to use as negatives (default: {DEFAULT_ABNORMAL_RATIO}, can be set via TRIPLET_ABNORMAL_RATIO env var)'
    )
    parser.add_argument(
        '--calibration-samples',
        type=int,
        default=DEFAULT_CALIBRATION_SAMPLES,
        help=f'Number of mel spectrograms to use for calibration (default: {DEFAULT_CALIBRATION_SAMPLES}, can be set via VALIDATION_CALIBRATION_SAMPLES env var)'
    )
    parser.add_argument(
        '--threshold-std-multiplier',
        type=float,
        default=DEFAULT_THRESHOLD_STD_MULTIPLIER,
        help=f'Multiplier for standard deviation when calculating anomaly threshold (default: {DEFAULT_THRESHOLD_STD_MULTIPLIER}, can be set via VALIDATION_THRESHOLD_STD_MULTIPLIER env var). Threshold = mean + multiplier * std'
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
                calibration_samples=args.calibration_samples,
                threshold_std_multiplier=args.threshold_std_multiplier,
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
            mlflow_run_id=args.mlflow_run_id,
            abnormal_ratio=args.abnormal_ratio,
            calibration_samples=args.calibration_samples,
            threshold_std_multiplier=args.threshold_std_multiplier
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
