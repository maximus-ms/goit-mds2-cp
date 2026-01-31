#!/usr/bin/env python3
"""
Training Script for Audio Anomaly Detection with MLflow Integration

This module provides training functionality with MLflow integration and early stopping.
It can be run as a standalone script or imported as a module for training.

Configuration:
    The module supports configuration via environment variables or .env file.
    See env.example for available configuration options.
    If python-dotenv is installed, .env file will be automatically loaded.

Usage as module:
    from train import train_mlflow
    dataset, history, timestamp, mlflow_model_path = train_mlflow(
        experiment_name="MyExperiment",
        epochs=30,
        batch_size=64
    )

Usage as script:
    python train.py [--experiment-name MyExperiment] [--epochs 30] [--batch-size 64]
"""

# Lazy import torch to avoid Jupyter kernel crashes
# Torch will be imported inside functions where it's needed
# This prevents multiprocessing issues when importing the module in Jupyter
import time
import os
import json
import argparse
import logging
import tempfile
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

# Import mlflow with error handling
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError as e:
    MLFLOW_AVAILABLE = False
    mlflow = None  # Create a dummy object to prevent NameError
    import warnings
    warnings.warn(f"MLflow not available: {e}. MLflow functionality will be disabled.")

# Lazy import local modules to avoid importing torch at module level
# These will be imported inside functions where needed
_MODEL_MODULE = None
_TRIPLET_DATASET_MODULE = None
_EARLY_STOPPING_MODULE = None

def _import_torch_modules():
    """Lazy import of torch and local modules to avoid Jupyter kernel crashes."""
    global _MODEL_MODULE, _TRIPLET_DATASET_MODULE, _EARLY_STOPPING_MODULE
    
    if _MODEL_MODULE is None:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        
        # Import local modules
        # Support both absolute and relative imports
        try:
            from ml.model import TinyAudioCNN, TinyAudioCNN_v2, TinyAudioCNN_v3, TinyAudioCNN_v4
            from ml.triplet_memory_dataset import TripletMemoryDataset
            from ml.early_stopping import EarlyStopping
        except ImportError:
            # Fallback to relative imports (when running as script from ml/ directory)
            from model import TinyAudioCNN, TinyAudioCNN_v2, TinyAudioCNN_v3, TinyAudioCNN_v4
            from triplet_memory_dataset import TripletMemoryDataset
            from early_stopping import EarlyStopping
        
        _MODEL_MODULE = {
            'torch': torch,
            'nn': nn,
            'optim': optim,
            'DataLoader': DataLoader,
            'TinyAudioCNN': TinyAudioCNN,
            'TinyAudioCNN_v2': TinyAudioCNN_v2,
            'TinyAudioCNN_v3': TinyAudioCNN_v3,
            'TinyAudioCNN_v4': TinyAudioCNN_v4,
            'TripletMemoryDataset': TripletMemoryDataset,
            'EarlyStopping': EarlyStopping
        }
    
    return _MODEL_MODULE

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging (must be before MinIO config to use logger)
# Only configure if logging hasn't been configured yet (e.g., in Jupyter)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

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

# Constants (can be overridden via environment variables)
DEFAULT_MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
DEFAULT_DATASET_FILE = os.getenv('DEFAULT_DATASET_FILE', None)
if DEFAULT_DATASET_FILE == '': DEFAULT_DATASET_FILE = None
DEFAULT_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'MIMII_Triplet_Training')
DEFAULT_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE', '64'))
DEFAULT_EPOCHS = int(os.getenv('TRAIN_EPOCHS', '30'))
DEFAULT_EMBEDDING_DIM = int(os.getenv('MODEL_EMBEDDING_DIM', '64'))
DEFAULT_MARGIN = float(os.getenv('TRAIN_MARGIN', '1.0'))
DEFAULT_SAMPLES_PER_EPOCH = int(os.getenv('TRIPLET_SAMPLES_PER_EPOCH', '5000'))
DEFAULT_LEARNING_RATE = float(os.getenv('TRAIN_LEARNING_RATE', '0.001'))
DEFAULT_MIN_LEARNING_RATE = float(os.getenv('TRAIN_MIN_LEARNING_RATE', '0.00001'))  # eta_min for CosineAnnealingWarmRestarts
DEFAULT_SCHEDULER_T0 = int(os.getenv('TRAIN_SCHEDULER_T0', '20'))  # T_0 for CosineAnnealingWarmRestarts
DEFAULT_SCHEDULER_T_MULT = int(os.getenv('TRAIN_SCHEDULER_T_MULT', '1'))  # T_mult for CosineAnnealingWarmRestarts
DEFAULT_SCHEDULER_RESTART_DECAY = float(os.getenv('TRAIN_SCHEDULER_RESTART_DECAY', '0.8'))  # Decay factor for restart LR
DEFAULT_WEIGHT_DECAY = float(os.getenv('TRAIN_WEIGHT_DECAY', '0.0001'))  # Weight decay for AdamW optimizer
DEFAULT_EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', '10'))
DEFAULT_EARLY_STOPPING_MIN_DELTA = float(os.getenv('EARLY_STOPPING_MIN_DELTA', '0.0'))
DEFAULT_EARLY_STOPPING_MIN_LOSS = os.getenv('EARLY_STOPPING_MIN_LOSS', None)
DEFAULT_EARLY_STOPPING_MIN_LOSS = float(DEFAULT_EARLY_STOPPING_MIN_LOSS) if DEFAULT_EARLY_STOPPING_MIN_LOSS else None
DEFAULT_EARLY_STOPPING_ENABLED = os.getenv('EARLY_STOPPING_ENABLED', 'true').lower() == 'true'
DEFAULT_EARLY_STOPPING_MIN_EPOCHS = int(os.getenv('EARLY_STOPPING_MIN_EPOCHS', '30'))
DEFAULT_SAVE_CHECKPOINTS = os.getenv('SAVE_CHECKPOINTS', 'false').lower() == 'true'
DEFAULT_MODEL_VERSION = os.getenv('MODEL_VERSION', '1')  # '1' or '2'
DEFAULT_LOSS_FUNCTION = os.getenv('TRAIN_LOSS_FUNCTION', 'triplet_loss')  # 'triplet_loss' or 'arcface'


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class ArcFaceLoss:
    """
    Angular Triplet Loss (ArcFace-style) for metric learning with triplet structure.
    
    This loss adds an angular margin to the feature space, forcing embeddings
    to be distributed on a hypersphere. This normalizes vectors and makes
    distances more interpretable.
    
    For triplet structure (anchor, positive, negative), we apply angular margin
    between anchor-positive and anchor-negative pairs.
    
    Formula (Angular Triplet Loss):
        L = max(0, cos(θ_an) - cos(θ_ap + m) + α)
    
    where:
        θ_ap = angle between anchor and positive
        θ_an = angle between anchor and negative
        m = angular margin (in radians)
        α = additional margin (typically 0.0 or small positive value)
    
    This is more suitable for triplet learning than standard ArcFace (which is
    designed for classification with class centers).
    """
    
    def __init__(self, margin: float = 0.5, alpha: float = 0.0):
        """
        Initialize Angular Triplet Loss.
        
        Args:
            margin: Angular margin in radians (default: 0.5, ~28.6 degrees)
            alpha: Additional margin for triplet loss (default: 0.0)
        """
        self.margin = margin
        self.alpha = alpha
    
    def __call__(self, anchor, positive, negative):
        """
        Compute Angular Triplet Loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
        
        Returns:
            Loss value (scalar tensor)
        """
        # Lazy import torch modules
        import torch
        import torch.nn.functional as F
        
        # Normalize embeddings to unit sphere
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        negative_norm = F.normalize(negative, p=2, dim=1)
        
        # Compute cosine similarities
        cos_ap = (anchor_norm * positive_norm).sum(dim=1)  # [batch_size]
        cos_an = (anchor_norm * negative_norm).sum(dim=1)  # [batch_size]
        
        # Clamp to valid range [-1, 1] for acos
        cos_ap = torch.clamp(cos_ap, -1.0 + 1e-7, 1.0 - 1e-7)
        cos_an = torch.clamp(cos_an, -1.0 + 1e-7, 1.0 - 1e-7)
        
        # Compute angles
        theta_ap = torch.acos(cos_ap)  # [batch_size]
        
        # Apply angular margin to positive (make it harder)
        theta_ap_margin = theta_ap + self.margin
        
        # Compute cosine with margin
        cos_ap_margin = torch.cos(theta_ap_margin)
        
        # Angular Triplet Loss: maximize cos(θ_ap + m) and minimize cos(θ_an)
        # Correct formula: L = max(0, cos(θ_ap + m) - cos(θ_an) + α)
        # We want: cos(θ_ap + m) > cos(θ_an) - α
        # This means: angle between anchor-positive should be SMALLER than anchor-negative
        # Note: cos decreases as angle increases, so we want cos_ap_margin > cos_an
        loss = torch.clamp(cos_ap_margin - cos_an + self.alpha, min=0.0)
        
        return loss.mean()


class CosineAnnealingWarmRestartsWithDecay:
    """
    Custom scheduler that wraps CosineAnnealingWarmRestarts and decays the maximum LR
    after each restart by a decay factor.
    
    This allows the learning rate to gradually decrease over multiple restart cycles,
    providing a balance between exploration (high LR) and fine-tuning (low LR).
    
    Args:
        optimizer: Optimizer to schedule
        T_0: Number of epochs for first restart cycle
        T_mult: Multiplier for restart period (1 = constant period, 2 = doubling period)
        eta_min: Minimum learning rate
        decay_factor: Factor to multiply max_lr after each restart (default: 0.8)
    """
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, decay_factor=0.8):
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        
        # Import torch.optim here to avoid module-level import
        import torch.optim as optim
        self.base_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        
        # Track restart epochs and current max LR
        self.last_epoch = -1
        self.current_max_lr = optimizer.param_groups[0]['lr']
        self.initial_lr = self.current_max_lr
    
    def step(self, epoch=None):
        """Step the scheduler and apply decay after restarts."""
        # Get current epoch
        if epoch is None:
            epoch = self.base_scheduler.last_epoch + 1
        
        # Calculate if this epoch will trigger a restart in the base scheduler
        # CosineAnnealingWarmRestarts restarts when T_cur >= T_i
        # For T_mult=1: restarts at epochs T_0, 2*T_0, 3*T_0, ...
        # The restart happens AFTER step(), so we check if (epoch + 1) will be a restart epoch
        is_restart = False
        next_epoch = epoch + 1
        
        if next_epoch > 0:
            # Calculate restart epochs based on T_0 and T_mult
            # CosineAnnealingWarmRestarts restarts when T_cur >= T_i
            # For T_mult=1: restarts at epochs T_0, 2*T_0, 3*T_0, ... (i.e., 20, 40, 60, ...)
            # But the restart happens AFTER step(), so we check if next_epoch will trigger restart
            restart_epochs = []
            if self.T_mult == 1:
                # Simple case: restarts at T_0, 2*T_0, 3*T_0, ...
                # But restart happens when T_cur resets, which is at epochs T_0, 2*T_0, etc.
                # Actually, looking at the base scheduler behavior, restart happens at epoch T_0+1, 2*T_0+1, etc.
                # Let's check if next_epoch is a multiple of T_0 (for T_mult=1)
                restart_epochs = [i * self.T_0 for i in range(1, (next_epoch // self.T_0) + 3)]
            else:
                # Complex case: T_0, T_0 + T_0*T_mult, T_0 + T_0*T_mult + T_0*T_mult^2, ...
                current_T = self.T_0
                cumulative = self.T_0
                while cumulative <= next_epoch + self.T_0 * 2:
                    restart_epochs.append(cumulative)
                    current_T *= self.T_mult
                    cumulative += current_T
            
            # Check if next epoch will be a restart (and we haven't processed it yet)
            # For T_mult=1, restart happens when epoch becomes a multiple of T_0
            if self.T_mult == 1:
                # Restart happens when epoch becomes T_0, 2*T_0, 3*T_0, etc.
                # But base scheduler does it at epoch T_0+1, 2*T_0+1, etc. (after step)
                # So we check if next_epoch == T_0 + 1, 2*T_0 + 1, etc.
                if (next_epoch - 1) % self.T_0 == 0 and next_epoch > self.T_0:
                    is_restart = True
            else:
                if next_epoch in restart_epochs and next_epoch != self.last_epoch + 1:
                    is_restart = True
        
        # Step the base scheduler first
        self.base_scheduler.step(epoch)
        
        # Check if a restart just happened by checking T_cur or LR jump
        # CosineAnnealingWarmRestarts resets T_cur to 0 when restart happens
        is_restart_detected = False
        current_lr = self.optimizer.param_groups[0]['lr']
        
        if epoch > 0:
            # Check if T_cur reset (indicating restart)
            if hasattr(self.base_scheduler, 'T_cur') and self.base_scheduler.T_cur < 1.0:
                # Check if LR is close to current_max_lr (within 5%), indicating restart
                if abs(current_lr - self.current_max_lr) / self.current_max_lr < 0.05:
                    is_restart_detected = True
        
        # If a restart just happened (after the first cycle), apply decay
        if is_restart_detected and epoch >= self.T_0:
            # Decay the current max LR
            self.current_max_lr *= self.decay_factor
            
            # Manually set the LR to the new decayed max_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_max_lr
                param_group['initial_lr'] = self.current_max_lr
            
            # Update the base scheduler's base_lrs for future calculations
            if hasattr(self.base_scheduler, 'base_lrs'):
                self.base_scheduler.base_lrs = [self.current_max_lr] * len(self.optimizer.param_groups)
            
            # Update base_scheduler's _last_lr to reflect the manual LR update
            if hasattr(self.base_scheduler, '_last_lr'):
                self.base_scheduler._last_lr = [self.current_max_lr] * len(self.optimizer.param_groups)
        
        self.last_epoch = epoch
    
    def get_last_lr(self):
        """Get the last computed learning rate."""
        # Return LR from optimizer to ensure we get the actual current LR
        # (which may have been manually set after restart)
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        """Return the state of the scheduler."""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            'current_max_lr': self.current_max_lr,
            'last_epoch': self.last_epoch,
            'decay_factor': self.decay_factor,
            'initial_lr': self.initial_lr,
        }
    
    def load_state_dict(self, state_dict):
        """Load the state of the scheduler."""
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.current_max_lr = state_dict['current_max_lr']
        self.last_epoch = state_dict['last_epoch']
        self.decay_factor = state_dict['decay_factor']
        self.initial_lr = state_dict.get('initial_lr', self.current_max_lr)


def train_mlflow(experiment_name: str = DEFAULT_EXPERIMENT_NAME,
                 run_name: Optional[str] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 epochs: int = DEFAULT_EPOCHS,
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                 margin: float = DEFAULT_MARGIN,
                 samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH,
                 lr: float = DEFAULT_LEARNING_RATE,
                 min_lr: float = DEFAULT_MIN_LEARNING_RATE,
                 scheduler_t0: int = DEFAULT_SCHEDULER_T0,
                 scheduler_t_mult: int = DEFAULT_SCHEDULER_T_MULT,
                 scheduler_restart_decay: float = DEFAULT_SCHEDULER_RESTART_DECAY,
                 weight_decay: float = DEFAULT_WEIGHT_DECAY,
                 dataset_file: Optional[str] = None,
                 mlflow_tracking_uri: Optional[str] = None,
                 skip_types: List[str] = None,
                 early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
                 early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
                 early_stopping_min_loss: Optional[float] = None,
                 early_stopping_enabled: bool = DEFAULT_EARLY_STOPPING_ENABLED,
                 early_stopping_min_epochs: int = DEFAULT_EARLY_STOPPING_MIN_EPOCHS,
                 save_checkpoints: bool = DEFAULT_SAVE_CHECKPOINTS,
                 model_version: str = DEFAULT_MODEL_VERSION,
                 loss_function: str = DEFAULT_LOSS_FUNCTION,
                 force_cpu: bool = False) -> Tuple:
    """
    Training function with MLflow integration and Early Stopping.
    
    All models and artifacts are saved to temporary files and uploaded to MLflow.
    No local files are persisted after training completes.
    
    Args:
        experiment_name: Name of MLflow experiment
        run_name: Name of specific run (None = auto-generated with timestamp)
        batch_size: Batch size
        epochs: Maximum number of epochs
        embedding_dim: Embedding vector dimension
        margin: Margin for Triplet Loss
        samples_per_epoch: Number of triplets per epoch
        lr: Initial learning rate (max_lr for CosineAnnealingWarmRestarts, default: 0.001)
        min_lr: Minimum learning rate for CosineAnnealingWarmRestarts (eta_min, default: 0.00001)
        scheduler_t0: Number of epochs for first restart cycle (T_0, default: 20)
        scheduler_t_mult: Multiplier for restart period (T_mult, default: 1)
        scheduler_restart_decay: Decay factor for restart LR (default: 0.8). Each restart multiplies max_lr by this factor
        dataset_file: Path to dataset .pt file (None = use DEFAULT_DATASET_FILE)
        mlflow_tracking_uri: MLflow tracking URI (None = use DEFAULT_MLFLOW_TRACKING_URI)
        skip_types: Machine types to skip (e.g., ['fan'] to exclude fan data)
        early_stopping_patience: Number of epochs without improvement before stopping
        early_stopping_min_delta: Minimum improvement to consider epoch as improvement
        early_stopping_min_loss: Minimum loss threshold to stop training (None = disabled)
        early_stopping_enabled: Enable/disable early stopping
        early_stopping_min_epochs: Minimum number of epochs before early stopping can trigger (default: 30)
        save_checkpoints: Enable/disable saving checkpoints every 5 epochs (default: False, can be set via SAVE_CHECKPOINTS env var)
        model_version: Model version to use ('1'/'v1' for TinyAudioCNN, '2'/'v2' for TinyAudioCNN_v2, '3'/'v3' for TinyAudioCNN_v3, '4'/'v4' for TinyAudioCNN_v4, default: '1')
        loss_function: Loss function to use ('triplet_loss' or 'arcface', default: 'triplet_loss')
        force_cpu: Force training on CPU (default: False)
    
    Returns:
        tuple: (dataset, history, timestamp, mlflow_model_path)
    """
    # Validate and normalize model_version
    # Support both old format ('v1', 'v2', 'v3') and new format ('1', '2', '3') for backward compatibility
    model_version = str(model_version).lower().strip()
    if model_version.startswith('v'):
        model_version = model_version[1:]  # Remove 'v' prefix if present
    
    if model_version not in ['1', '2', '3', '4']:
        raise ValueError(f"model_version must be '1', '2', '3', or '4' (or 'v1'/'v2'/'v3'/'v4'), got '{model_version}'")
    
    # Lazy import torch modules to avoid Jupyter kernel crashes
    modules = _import_torch_modules()
    torch = modules['torch']
    nn = modules['nn']
    optim = modules['optim']
    DataLoader = modules['DataLoader']
    TinyAudioCNN = modules['TinyAudioCNN']
    TinyAudioCNN_v2 = modules['TinyAudioCNN_v2']
    TinyAudioCNN_v3 = modules['TinyAudioCNN_v3']
    TinyAudioCNN_v4 = modules['TinyAudioCNN_v4']
    TripletMemoryDataset = modules['TripletMemoryDataset']
    EarlyStopping = modules['EarlyStopping']
    
    # Select model class based on version
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
    
    # Check if MLflow is available
    if not MLFLOW_AVAILABLE:
        raise ImportError(
            "MLflow is not installed. Please install it with: pip install mlflow"
        )
    
    # Use defaults from environment if not provided
    if dataset_file is None:
        dataset_file = DEFAULT_DATASET_FILE
    if dataset_file is None:
        raise ValueError(
            "dataset_file is required. Please provide it as an argument or set "
            "DEFAULT_DATASET_FILE environment variable."
        )
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = DEFAULT_MLFLOW_TRACKING_URI
    if skip_types is None:
        skip_types = []
    
    # Setup file logging for training
    log_file_handler = None
    log_file_path = None
    
    try:
        # Extract dataset name from path for log file naming
        dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
        # Replace spaces and special characters with underscores for filename safety
        dataset_name = dataset_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        
        # Create temporary log file with dataset name
        timestamp_log = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = tempfile.NamedTemporaryFile(
            mode='w',
            prefix=f'{dataset_name}_training_',
            suffix=f'_{timestamp_log}.log',
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
        logger.info(f"Training log file: {log_file_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create log file handler: {e}")
        log_file_handler = None
    
    # Create temporary directory for all artifacts (will be cleaned up automatically)
    temp_dir = tempfile.mkdtemp(prefix="mlflow_train_")
    logger.debug(f"Created temporary directory: {temp_dir}")
    temp_files_to_cleanup = []  # Track temp files for cleanup
    mlflow_model_path = None  # Initialize in case of error
    
    try:
        # MLflow setup
        logger.info(f"Configuring MLflow: {mlflow_tracking_uri}")
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        # Create run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name is None:
            run_name = f"train_{timestamp}"
        
        # Start MLflow run
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_name=run_name):
            # 1. Hardware setup
            if force_cpu:
                device = torch.device("cpu")
                logger.info("🚀 CPU activated!")
                device_name = "CPU"
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("🚀 Apple Silicon (MPS) activated!")
                device_name = "MPS"
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("🚀 CUDA GPU activated!")
                device_name = "CUDA"
            else:
                device = torch.device("cpu")
                logger.warning("⚠️ Warning: Training on CPU. This may be slow.")
                device_name = "CPU"
            
            # Log model and hyperparameters
            mlflow.log_params({
                "batch_size": batch_size,
                "epochs": epochs,
                "samples_per_epoch": samples_per_epoch,
                "embedding_dim": embedding_dim,
                "margin": margin,
                "learning_rate": lr,
                "min_learning_rate": min_lr,
                "optimizer": "AdamW",
                "weight_decay": weight_decay,
                "loss_function": loss_function,
                "scheduler": "CosineAnnealingWarmRestartsWithDecay",
                "scheduler_T0": scheduler_t0,
                "scheduler_T_mult": scheduler_t_mult,
                "scheduler_eta_min": min_lr,
                "scheduler_restart_decay": scheduler_restart_decay,
                "device": device_name,
                "dataset_file": dataset_file,
                "skip_types": f"{skip_types}",
                # Early Stopping parameters
                "early_stopping_enabled": early_stopping_enabled,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "early_stopping_min_loss": early_stopping_min_loss if early_stopping_min_loss else "None",
                "early_stopping_min_epochs": early_stopping_min_epochs,
                "save_checkpoints": save_checkpoints,
            })
            
            # Log tags
            mlflow.set_tags({
                "model_type": model_name,
                "model_version": model_version,  # Store as '1' or '2'
                "task": "triplet_learning",
                "dataset": "MIMII",
            })
            
            # Log model version as parameter (store as '1' or '2')
            mlflow.log_param("model_version", model_version)

            # 2. Dataset initialization
            logger.info("⏳ Loading data into memory...")
            if not os.path.exists(dataset_file):
                raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
            
            dataset = TripletMemoryDataset(
                dataset_file,
                samples_per_epoch=samples_per_epoch,
                sample_rate=16000,
                skip_types=skip_types
            )
            
            train_loader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0,
                pin_memory=False
            )
            
            # Log dataset information
            mlflow.log_param("dataset_size", len(dataset))
            mlflow.log_param("num_batches_per_epoch", len(train_loader))

            # 3. Model initialization
            logger.info(f"📦 Using model: {model_name} (version {model_version})")
            model = ModelClass(embedding_dim=embedding_dim).to(device)
            
            # Count model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_params({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            })
            logger.info(f"📊 Model has {total_params:,} parameters ({trainable_params:,} trainable)")
            
            # 4. Loss function and Optimizer
            loss_function_lower = loss_function.lower()
            if loss_function_lower == 'arcface':
                criterion = ArcFaceLoss(margin=margin, alpha=0.0)
                logger.info(f"📊 Using Angular Triplet Loss (ArcFace-style, angular margin={margin})")
            elif loss_function_lower == 'triplet_loss' or loss_function_lower == 'tripletmarginloss':
                criterion = nn.TripletMarginLoss(margin=margin, p=2)
                logger.info(f"📊 Using TripletMarginLoss (margin={margin})")
            else:
                raise ValueError(f"Unknown loss function: {loss_function}. Supported: 'triplet_loss', 'arcface'")
            # Use AdamW instead of Adam for better weight decay handling
            # AdamW applies weight decay directly to weights (decoupled), not to gradients
            # This leads to better generalization and more stable training
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            # Cosine Annealing with Warm Restarts: periodically restarts LR to max
            # This helps escape local minima and improves convergence
            # T_0: number of epochs for first restart cycle
            # T_mult: multiplier for restart period (1 = constant period, 2 = doubling period)
            scheduler = CosineAnnealingWarmRestartsWithDecay(
                optimizer,
                T_0=scheduler_t0,
                T_mult=scheduler_t_mult,
                eta_min=min_lr,
                decay_factor=scheduler_restart_decay
            )

            logger.info(f"🏁 Starting training for {epochs} epochs...")
            logger.info(f"📈 Learning Rate Schedule: CosineAnnealingWarmRestartsWithDecay (T_0={scheduler_t0}, T_mult={scheduler_t_mult}, eta_min={min_lr:.2e}, decay={scheduler_restart_decay})")
            
            # Log checkpoint saving status
            if save_checkpoints:
                logger.info("💾 Checkpoint saving enabled (every 5 epochs)")
            else:
                logger.info("💾 Checkpoint saving disabled")
            
            # Initialize Early Stopping
            early_stopping = None
            if early_stopping_enabled:
                early_stopping = EarlyStopping(
                    patience=early_stopping_patience,
                    min_delta=early_stopping_min_delta,
                    min_loss=early_stopping_min_loss if early_stopping_min_loss is not None else DEFAULT_EARLY_STOPPING_MIN_LOSS,
                    mode='min',
                    verbose=True,
                    restore_best_weights=True,
                    min_epochs=early_stopping_min_epochs
                )
                logger.info(f"🛡️ Early Stopping activated: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}, min_epochs={early_stopping_min_epochs}")
                if early_stopping_min_loss:
                    logger.info(f"   Minimum loss threshold: {early_stopping_min_loss}")
            
            best_loss = float('inf')
            history = []
            best_model_path = None
            stopped_early = False

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                start_time = time.time()
                
                # Diagnostic: track loss distribution for first few epochs
                if epoch < 3 and loss_function_lower == 'arcface':
                    loss_values = []
                    zero_loss_count = 0
                    total_samples = 0
                
                for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
                    anchor = anchor.to(device)
                    positive = positive.to(device)
                    negative = negative.to(device)
                    
                    optimizer.zero_grad()
                    emb_a = model(anchor)
                    emb_p = model(positive)
                    emb_n = model(negative)
                    loss = criterion(emb_a, emb_p, emb_n)
                    
                    # Diagnostic: track loss distribution for first few epochs
                    if epoch < 3 and loss_function_lower == 'arcface':
                        loss_batch = loss.item()
                        loss_values.append(loss_batch)
                        # For Angular Triplet Loss, check individual triplet losses
                        if hasattr(criterion, 'margin'):
                            # Compute individual losses to check distribution
                            import torch.nn.functional as F
                            anchor_norm = F.normalize(emb_a, p=2, dim=1)
                            positive_norm = F.normalize(emb_p, p=2, dim=1)
                            negative_norm = F.normalize(emb_n, p=2, dim=1)
                            cos_ap = (anchor_norm * positive_norm).sum(dim=1)
                            cos_an = (anchor_norm * negative_norm).sum(dim=1)
                            cos_ap = torch.clamp(cos_ap, -1.0 + 1e-7, 1.0 - 1e-7)
                            cos_an = torch.clamp(cos_an, -1.0 + 1e-7, 1.0 - 1e-7)
                            theta_ap = torch.acos(cos_ap)
                            theta_ap_margin = theta_ap + criterion.margin
                            cos_ap_margin = torch.cos(theta_ap_margin)
                            individual_losses = torch.clamp(cos_ap_margin - cos_an + criterion.alpha, min=0.0)
                            zero_loss_count += (individual_losses == 0.0).sum().item()
                            total_samples += individual_losses.shape[0]
                    
                    loss.backward()
                    
                    # Diagnostic: check gradients for first batch of first epoch
                    if epoch == 0 and batch_idx == 0 and loss_function_lower == 'arcface':
                        total_grad_norm = 0.0
                        param_count = 0
                        for param in model.parameters():
                            if param.grad is not None:
                                param_grad_norm = param.grad.data.norm(2).item()
                                total_grad_norm += param_grad_norm ** 2
                                param_count += 1
                        total_grad_norm = total_grad_norm ** 0.5
                        logger.info(f"  📊 First batch diagnostics:")
                        logger.info(f"     Loss: {loss.item():.6f}")
                        logger.info(f"     Gradient norm: {total_grad_norm:.6f}")
                        if total_samples > 0:
                            zero_loss_ratio = zero_loss_count / total_samples
                            logger.info(f"     Triplets with loss=0: {zero_loss_count}/{total_samples} ({zero_loss_ratio:.1%})")
                    
                    optimizer.step()
                    running_loss += loss.item()
                
                # Diagnostic: log loss distribution for first few epochs
                if epoch < 3 and loss_function_lower == 'arcface' and len(loss_values) > 0:
                    import numpy as np
                    loss_array = np.array(loss_values)
                    logger.info(f"  📊 Epoch {epoch+1} loss distribution:")
                    logger.info(f"     Mean: {loss_array.mean():.6f}, Std: {loss_array.std():.6f}")
                    logger.info(f"     Min: {loss_array.min():.6f}, Max: {loss_array.max():.6f}")
                    if total_samples > 0:
                        logger.info(f"     Triplets with loss=0: {zero_loss_count}/{total_samples} ({zero_loss_count/total_samples:.1%})")

                # End of epoch
                avg_loss = running_loss / len(train_loader)
                duration = time.time() - start_time
                current_lr = scheduler.get_last_lr()[0]
                history.append(avg_loss)
                
                # Log metrics for each epoch
                mlflow.log_metrics({
                    "train_loss": avg_loss,
                    "learning_rate": current_lr,
                    "epoch_time": duration,
                }, step=epoch)
                
                logger.info(f"Epoch [{epoch+1}/{epochs}] | "
                      f"Loss: {avg_loss:.6f} | "
                      f"LR: {current_lr:.1e} | "
                      f"Time: {duration:.1f}s")
                
                # Check Early Stopping (BEFORE scheduler.step and model saving)
                if early_stopping is not None:
                    if early_stopping(avg_loss, epoch, model):
                        stopped_early = True
                        logger.warning(f"\n🛑 Training stopped early at epoch {epoch+1}/{epochs}")
                        # Restore best weights
                        early_stopping.restore_weights(model)
                        # Save best model to temporary file
                        best_model_path = os.path.join(temp_dir, f"{timestamp}_best_model.pth")
                        torch.save(model.state_dict(), best_model_path)
                        temp_files_to_cleanup.append(best_model_path)
                        
                        # Log early stopping information
                        es_info = early_stopping.get_info()
                        mlflow.log_metrics({
                            "early_stopped": 1,
                            "early_stop_epoch": epoch + 1,
                            "early_stop_best_epoch": es_info['best_epoch'] + 1,
                        })
                        mlflow.log_params({
                            "early_stop_reason": "patience" if es_info['counter'] >= early_stopping_patience else "min_loss",
                        })
                        break
                
                scheduler.step()
                
                # Save "best" model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_model_path = os.path.join(temp_dir, f"{timestamp}_best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                    if best_model_path not in temp_files_to_cleanup:
                        temp_files_to_cleanup.append(best_model_path)
                    mlflow.log_metric("best_loss", best_loss, step=epoch)
                    
                # Save checkpoint (if enabled)
                if save_checkpoints and (epoch + 1) % 5 == 0:
                    checkpoint_path = os.path.join(temp_dir, f"{timestamp}_checkpoint_epoch_{epoch+1}.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    temp_files_to_cleanup.append(checkpoint_path)
                    mlflow.log_artifact(checkpoint_path, "checkpoints")

            # Final training information
            if stopped_early:
                logger.warning("\n🛑 Training stopped early (Early Stopping)")
                if early_stopping is not None:
                    es_info = early_stopping.get_info()
                    logger.info(f"   Best result: {es_info['best_score']:.6f} at epoch {es_info['best_epoch']+1}")
                    logger.info(f"   Last result: {history[-1]:.6f} at epoch {len(history)}")
            else:
                logger.info("🎉 Training completed!")
            
            logger.info(f"Minimum Loss: {best_loss:.4f}")
            logger.info(f"Completed epochs: {len(history)}/{epochs}")
            
            # Save final version to temporary file
            final_model_path = os.path.join(temp_dir, f"{timestamp}_final_model.pth")
            torch.save(model.state_dict(), final_model_path)
            temp_files_to_cleanup.append(final_model_path)
            
            # Log final metrics
            final_metrics = {
                "final_loss": history[-1],
                "best_loss": best_loss,
                "total_epochs": len(history),
                "early_stopped": 1 if stopped_early else 0,
            }
            
            if early_stopping is not None and stopped_early:
                es_info = early_stopping.get_info()
                final_metrics.update({
                    "early_stop_best_epoch": es_info['best_epoch'] + 1,
                    "early_stop_best_score": es_info['best_score'],
                })
            
            mlflow.log_metrics(final_metrics)
            
            # Save models as artifacts
            if best_model_path and os.path.exists(best_model_path):
                mlflow.log_artifact(best_model_path, "models")
            if os.path.exists(final_model_path):
                mlflow.log_artifact(final_model_path, "models")
            
            # Save loss plot
            loss_plot_path = None
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 6))
                plt.plot(history)
                plt.title("Training Loss")
                plt.xlabel("Epoch")
                plt.ylabel(f"Loss ({loss_function})")
                plt.grid(True)
                if stopped_early and early_stopping is not None:
                    es_info = early_stopping.get_info()
                    plt.axvline(es_info['best_epoch'], color='green', linestyle='--', 
                               label=f"Best epoch ({es_info['best_epoch']+1})")
                    plt.axvline(len(history)-1, color='red', linestyle='--', 
                               label=f"Stopped ({len(history)})")
                plt.legend()
                loss_plot_path = os.path.join(temp_dir, f"{timestamp}_training_loss.png")
                plt.savefig(loss_plot_path)
                plt.close()
                temp_files_to_cleanup.append(loss_plot_path)
                mlflow.log_artifact(loss_plot_path, "plots")
                logger.info("Training loss plot saved to MLflow")
            except ImportError:
                logger.warning("matplotlib not available. Skipping plot generation.")
            except Exception as e:
                logger.error(f"Failed to save plot: {e}")
            
            # Save history as artifact
            history_json_path = os.path.join(temp_dir, f"{timestamp}_history.json")
            with open(history_json_path, 'w') as f:
                json.dump({"loss_history": history, "stopped_early": stopped_early}, f)
            temp_files_to_cleanup.append(history_json_path)
            mlflow.log_artifact(history_json_path, "history")
            
            run_id = mlflow.active_run().info.run_id
            logger.info(f"📊 MLflow run completed: {mlflow.get_tracking_uri()}")
            logger.info(f"Run ID: {run_id}")

            # Form MLflow path to best model
            if best_model_path and os.path.exists(best_model_path):
                mlflow_model_path = f"runs:/{run_id}/models/{os.path.basename(best_model_path)}"
                logger.info(f"📦 MLflow model path: {mlflow_model_path}")
            else:
                mlflow_model_path = None
                logger.warning("⚠️ Best model not found")

            # Save training log file as artifact
            if log_file_path and os.path.exists(log_file_path):
                try:
                    mlflow.log_artifact(log_file_path, "logs")
                    logger.info(f"Training log file saved to MLflow artifacts: logs/{os.path.basename(log_file_path)}")
                except Exception as e:
                    logger.warning(f"Failed to save log file to MLflow: {e}")
            
    
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
        
        # Cleanup temporary files (guaranteed to run even if error occurs)
        logger.debug("Cleaning up temporary files...")
        for temp_file in temp_files_to_cleanup:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
        
        # Remove temporary directory
        try:
            if os.path.exists(temp_dir):
                # Try to remove directory (will fail if not empty)
                try:
                    os.rmdir(temp_dir)
                except OSError:
                    # Directory not empty, remove all files first
                    import shutil
                    shutil.rmtree(temp_dir)
                logger.debug(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")
    
    return dataset, history, timestamp, mlflow_model_path


def test_model_mlflow(
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    batch_size: int = 2,
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Test the model with MLflow integration for automatic artifact logging.
    
    This function automatically configures MLflow, runs model tests, and logs
    all results, metrics, and artifacts to MLflow (with MinIO/S3 support).
    Minimal overhead - only creates a small dummy model for testing.
    
    Args:
        embedding_dim: Dimension of embedding vector
        batch_size: Batch size for test input
        tracking_uri: MLflow tracking URI (defaults to MLFLOW_TRACKING_URI env var)
        experiment_name: MLflow experiment name (defaults to "Model_Testing")
        run_name: Optional name for this MLflow run
    
    Returns:
        Dictionary with test results including 'success', 'model_info', 'test_metrics', 'mlflow_run_id'
    """
    # Lazy import torch modules to avoid Jupyter kernel crashes
    modules = _import_torch_modules()
    torch = modules['torch']
    TinyAudioCNN = modules['TinyAudioCNN']
    
    logger.info("🧪 Starting model test with MLflow integration...")
    
    # Use defaults from environment if not provided
    tracking_uri = tracking_uri or DEFAULT_MLFLOW_TRACKING_URI
    experiment_name = experiment_name or "Model_Testing"
    
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    
    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new MLflow experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name} (ID: {experiment_id})")
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.warning(f"Could not set MLflow experiment: {e}. Continuing without experiment.")
        experiment_id = None
    
    # Generate run name if not provided
    if not run_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"model_test_{embedding_dim}d_{timestamp}"
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Started MLflow run: {run_name} (ID: {run.info.run_id})")
        
        try:
            # Create dummy input: [Batch, 1, 64 (mel), 32 (time)]
            # This simulates 1 second of audio with hop_length=512
            dummy_input = torch.randn(batch_size, 1, 64, 32)
            logger.info(f"Input shape: {dummy_input.shape}")
            
            # Initialize model
            model = TinyAudioCNN(embedding_dim=embedding_dim)
            logger.info(f"Model initialized with embedding_dim={embedding_dim}")
            
            # Forward pass
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            logger.info(f"Output shape: {output.shape}")
            
            # Verify output properties
            # 1. Check output shape
            expected_shape = (batch_size, embedding_dim)
            shape_valid = output.shape == expected_shape
            if not shape_valid:
                logger.error(f"Output shape mismatch! Expected {expected_shape}, got {output.shape}")
                mlflow.log_metric("test_success", 0)
                mlflow.log_metric("shape_valid", 0)
                return {
                    'success': False,
                    'model_info': model.get_model_info(),
                    'test_metrics': {'shape_valid': False},
                    'mlflow_run_id': run.info.run_id
                }
            
            # 2. Check normalization (L2 norm should be ~1.0 for each vector)
            norms = torch.norm(output, p=2, dim=1)
            avg_norm = norms.mean().item()
            min_norm = norms.min().item()
            max_norm = norms.max().item()
            std_norm = norms.std().item()
            
            logger.info(f"Vector norms - Mean: {avg_norm:.6f}, Min: {min_norm:.6f}, Max: {max_norm:.6f}, Std: {std_norm:.6f}")
            
            # Allow small tolerance for floating point errors
            tolerance = 1e-5
            norm_valid = abs(avg_norm - 1.0) <= tolerance
            if not norm_valid:
                logger.warning(f"Average norm ({avg_norm:.6f}) deviates from 1.0 by more than {tolerance}")
            
            # 3. Get model statistics
            params = model.count_parameters()
            logger.info("Model parameters:")
            logger.info(f"  Total: {params['total']:,}")
            logger.info(f"  Trainable: {params['trainable']:,}")
            logger.info(f"  Non-trainable: {params['non_trainable']:,}")
            
            # 4. Get model info
            model_info = model.get_model_info()
            logger.info("Model information:")
            for key, value in model_info.items():
                logger.info(f"  {key}: {value}")
            
            success = shape_valid and norm_valid
            
            # Log parameters
            mlflow.log_param("embedding_dim", embedding_dim)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("test_type", "model_verification")
            
            # Log metrics
            mlflow.log_metric("test_success", 1 if success else 0)
            mlflow.log_metric("shape_valid", 1 if shape_valid else 0)
            mlflow.log_metric("norm_valid", 1 if norm_valid else 0)
            mlflow.log_metric("avg_norm", avg_norm)
            mlflow.log_metric("min_norm", min_norm)
            mlflow.log_metric("max_norm", max_norm)
            mlflow.log_metric("std_norm", std_norm)
            mlflow.log_metric("norm_tolerance", tolerance)
            
            # Log model parameters
            mlflow.log_param("total_parameters", params['total'])
            mlflow.log_param("trainable_parameters", params['trainable'])
            mlflow.log_param("non_trainable_parameters", params['non_trainable'])
            
            # Log model info as artifact (JSON)
            model_info_json = {
                k: v for k, v in model_info.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(model_info_json, f, indent=2)
                temp_path = f.name
            
            try:
                mlflow.log_artifact(temp_path, "model_info.json")
                logger.info("Logged model info to MLflow artifacts")
            finally:
                os.unlink(temp_path)
            
            # Log test metrics as artifact (JSON)
            test_metrics = {
                'shape_valid': shape_valid,
                'norm_valid': norm_valid,
                'avg_norm': avg_norm,
                'min_norm': min_norm,
                'max_norm': max_norm,
                'std_norm': std_norm,
                'norm_tolerance': tolerance,
                'input_shape': list(dummy_input.shape),
                'output_shape': list(output.shape),
                'expected_shape': list(expected_shape)
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(test_metrics, f, indent=2)
                temp_path = f.name
            
            try:
                mlflow.log_artifact(temp_path, "test_metrics.json")
                logger.info("Logged test metrics to MLflow artifacts")
            finally:
                os.unlink(temp_path)
            
            # Log the model state dict
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    model_path = os.path.join(temp_dir, "model.pt")
                    torch.save(model.state_dict(), model_path)
                    mlflow.log_artifact(model_path, "model")
                    logger.info("Logged model state dict to MLflow artifacts")
            except Exception as e:
                logger.warning(f"Could not log model to MLflow: {e}")
            
            if success:
                logger.info("✅ All tests passed!")
            else:
                logger.warning("⚠️ Some tests failed or produced warnings")
            
            return {
                'success': success,
                'model_info': model_info,
                'test_metrics': test_metrics,
                'mlflow_run_id': run.info.run_id,
                'mlflow_run_name': run_name,
                'mlflow_experiment_name': experiment_name
            }
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}", exc_info=True)
            mlflow.log_metric("test_success", 0)
            mlflow.log_param("test_error", str(e))
            return {
                'success': False,
                'error': str(e),
                'model_info': {},
                'test_metrics': {},
                'mlflow_run_id': run.info.run_id
            }


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Train TinyAudioCNN model with MLflow integration'
    )
    parser.add_argument(
        '--experiment-name', '-e',
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help=f'MLflow experiment name (default: {DEFAULT_EXPERIMENT_NAME})'
    )
    parser.add_argument(
        '--run-name', '-r',
        type=str,
        default=None,
        help='MLflow run name (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help=f'Path to dataset .pt file (default: {DEFAULT_DATASET_FILE})'
    )
    parser.add_argument(
        '--mlflow-uri',
        type=str,
        default=None,
        help=f'MLflow tracking URI (default: {DEFAULT_MLFLOW_TRACKING_URI})'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})'
    )
    parser.add_argument(
        '--epochs', '-n',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Number of epochs (default: {DEFAULT_EPOCHS})'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f'Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})'
    )
    parser.add_argument(
        '--loss-function',
        type=str,
        default=DEFAULT_LOSS_FUNCTION,
        choices=['triplet_loss', 'arcface'],
        help=f'Loss function to use: "triplet_loss" or "arcface" (default: {DEFAULT_LOSS_FUNCTION})'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=DEFAULT_MARGIN,
        help=f'Triplet loss margin or ArcFace angular margin (default: {DEFAULT_MARGIN})'
    )
    parser.add_argument(
        '--samples-per-epoch',
        type=int,
        default=DEFAULT_SAMPLES_PER_EPOCH,
        help=f'Samples per epoch (default: {DEFAULT_SAMPLES_PER_EPOCH})'
    )
    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f'Initial learning rate (max_lr for CosineAnnealingWarmRestarts, default: {DEFAULT_LEARNING_RATE})'
    )
    parser.add_argument(
        '--min-learning-rate', '--min-lr',
        type=float,
        default=DEFAULT_MIN_LEARNING_RATE,
        help=f'Minimum learning rate for CosineAnnealingWarmRestarts (eta_min, default: {DEFAULT_MIN_LEARNING_RATE}, can be set via TRAIN_MIN_LEARNING_RATE env var)'
    )
    parser.add_argument(
        '--scheduler-t0',
        type=int,
        default=DEFAULT_SCHEDULER_T0,
        help=f'Number of epochs for first restart cycle in CosineAnnealingWarmRestarts (T_0, default: {DEFAULT_SCHEDULER_T0}, can be set via TRAIN_SCHEDULER_T0 env var)'
    )
    parser.add_argument(
        '--scheduler-t-mult',
        type=int,
        default=DEFAULT_SCHEDULER_T_MULT,
        help=f'Multiplier for restart period in CosineAnnealingWarmRestarts (T_mult, default: {DEFAULT_SCHEDULER_T_MULT}, can be set via TRAIN_SCHEDULER_T_MULT env var). Use 1 for constant period, 2 for doubling period'
    )
    parser.add_argument(
        '--scheduler-restart-decay',
        type=float,
        default=DEFAULT_SCHEDULER_RESTART_DECAY,
        help=f'Decay factor for restart LR in CosineAnnealingWarmRestartsWithDecay (default: {DEFAULT_SCHEDULER_RESTART_DECAY}, can be set via TRAIN_SCHEDULER_RESTART_DECAY env var). Each restart multiplies max_lr by this factor'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help=f'Weight decay (L2 regularization) for AdamW optimizer (default: {DEFAULT_WEIGHT_DECAY}, can be set via TRAIN_WEIGHT_DECAY env var)'
    )
    parser.add_argument(
        '--skip-types',
        type=str,
        nargs='+',
        default=[],
        help='Machine types to skip (e.g., --skip-types fan pump)'
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help=f'Early stopping patience (default: {DEFAULT_EARLY_STOPPING_PATIENCE})'
    )
    parser.add_argument(
        '--early-stopping-min-delta',
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help=f'Early stopping min delta (default: {DEFAULT_EARLY_STOPPING_MIN_DELTA})'
    )
    parser.add_argument(
        '--early-stopping-min-loss',
        type=float,
        default=None,
        help='Early stopping minimum loss threshold (default: None)'
    )
    parser.add_argument(
        '--early-stopping-min-epochs',
        type=int,
        default=DEFAULT_EARLY_STOPPING_MIN_EPOCHS,
        help=f'Minimum number of epochs before early stopping can trigger (default: {DEFAULT_EARLY_STOPPING_MIN_EPOCHS}, can be set via EARLY_STOPPING_MIN_EPOCHS env var)'
    )
    parser.add_argument(
        '--no-early-stopping',
        action='store_true',
        help='Disable early stopping'
    )
    parser.add_argument(
        '--save-checkpoints',
        action='store_true',
        default=DEFAULT_SAVE_CHECKPOINTS,
        help=f'Enable saving checkpoints every 5 epochs (default: {DEFAULT_SAVE_CHECKPOINTS}, can be set via SAVE_CHECKPOINTS env var)'
    )
    parser.add_argument(
        '--model-version',
        type=str,
        choices=['1', '2', '3', '4', 'v1', 'v2', 'v3', 'v4'],
        default=DEFAULT_MODEL_VERSION,
        help=f'Model version to use: 1 (TinyAudioCNN), 2 (TinyAudioCNN_v2, optimized), 3 (TinyAudioCNN_v3, MobileNetV2 style, ~56k params), or 4 (TinyAudioCNN_v4, MBConv+SE, ~210k params) (default: {DEFAULT_MODEL_VERSION}, can also be set via MODEL_VERSION env var). Supports both formats: "1"/"2"/"3"/"4" or "v1"/"v2"/"v3"/"v4"'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run model test with MLflow integration (automatically configured, minimal overhead)'
    )
    
    args = parser.parse_args()
    
    # Handle test mode
    if args.test:
        try:
            logger.info("Running model test with MLflow integration...")
            test_results = test_model_mlflow(
                embedding_dim=args.embedding_dim,
                batch_size=args.batch_size,
                tracking_uri=args.mlflow_uri,
                experiment_name=args.experiment_name if args.experiment_name != DEFAULT_EXPERIMENT_NAME else None,
                run_name=args.run_name
            )
            
            if test_results.get('success'):
                logger.info("✅ Test completed successfully!")
                if 'mlflow_run_id' in test_results:
                    logger.info(f"MLflow run ID: {test_results['mlflow_run_id']}")
                    logger.info(f"MLflow experiment: {test_results.get('mlflow_experiment_name', 'N/A')}")
                return 0
            else:
                logger.error("❌ Test failed!")
                if 'error' in test_results:
                    logger.error(f"Error: {test_results['error']}")
                return 1
        except Exception as e:
            logger.error(f"Error during testing: {e}", exc_info=True)
            return 1
    
    # Normal training mode
    try:
        dataset, history, timestamp, mlflow_model_path = train_mlflow(
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            batch_size=args.batch_size,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            samples_per_epoch=args.samples_per_epoch,
            lr=args.learning_rate,
            min_lr=args.min_learning_rate,
            scheduler_t0=args.scheduler_t0,
            scheduler_t_mult=args.scheduler_t_mult,
            scheduler_restart_decay=args.scheduler_restart_decay,
            weight_decay=args.weight_decay,
            dataset_file=args.dataset,
            mlflow_tracking_uri=args.mlflow_uri,
            skip_types=args.skip_types,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_min_loss=args.early_stopping_min_loss,
            early_stopping_enabled=not args.no_early_stopping,
            early_stopping_min_epochs=args.early_stopping_min_epochs,
            save_checkpoints=args.save_checkpoints,
            model_version=args.model_version,
            loss_function=args.loss_function,
        )
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"MLflow model path: {mlflow_model_path}")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please check the dataset file path.")
        return 1
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
