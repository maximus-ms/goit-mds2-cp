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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
import argparse
import logging
import tempfile
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import mlflow

# Import local modules
# Support both absolute and relative imports
try:
    from ml.model import TinyAudioCNN
    from ml.triplet_memory_dataset import TripletMemoryDataset
    from ml.early_stopping import EarlyStopping
except ImportError:
    # Fallback to relative imports (when running as script from ml/ directory)
    from model import TinyAudioCNN
    from triplet_memory_dataset import TripletMemoryDataset
    from early_stopping import EarlyStopping

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configure logging (must be before MinIO config to use logger)
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
DEFAULT_DATASET_FILE = os.getenv('DEFAULT_DATASET_FILE', 'dataset.pt')
DEFAULT_MODELS_DIR = os.getenv('MODELS_DIR', 'models')
DEFAULT_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'MIMII_Triplet_Training')
DEFAULT_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE', '64'))
DEFAULT_EPOCHS = int(os.getenv('TRAIN_EPOCHS', '30'))
DEFAULT_EMBEDDING_DIM = int(os.getenv('MODEL_EMBEDDING_DIM', '64'))
DEFAULT_MARGIN = float(os.getenv('TRAIN_MARGIN', '1.0'))
DEFAULT_SAMPLES_PER_EPOCH = int(os.getenv('TRIPLET_SAMPLES_PER_EPOCH', '5000'))
DEFAULT_LEARNING_RATE = float(os.getenv('TRAIN_LEARNING_RATE', '0.001'))
DEFAULT_EARLY_STOPPING_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', '10'))
DEFAULT_EARLY_STOPPING_MIN_DELTA = float(os.getenv('EARLY_STOPPING_MIN_DELTA', '0.0'))
DEFAULT_EARLY_STOPPING_MIN_LOSS = os.getenv('EARLY_STOPPING_MIN_LOSS', None)
DEFAULT_EARLY_STOPPING_MIN_LOSS = float(DEFAULT_EARLY_STOPPING_MIN_LOSS) if DEFAULT_EARLY_STOPPING_MIN_LOSS else None
DEFAULT_EARLY_STOPPING_ENABLED = os.getenv('EARLY_STOPPING_ENABLED', 'true').lower() == 'true'


def train_mlflow(experiment_name: str = DEFAULT_EXPERIMENT_NAME,
                 run_name: Optional[str] = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 epochs: int = DEFAULT_EPOCHS,
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                 margin: float = DEFAULT_MARGIN,
                 samples_per_epoch: int = DEFAULT_SAMPLES_PER_EPOCH,
                 lr: float = DEFAULT_LEARNING_RATE,
                 dataset_file: Optional[str] = None,
                 models_dir: Optional[str] = None,
                 mlflow_tracking_uri: Optional[str] = None,
                 skip_types: List[str] = None,
                 early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
                 early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
                 early_stopping_min_loss: Optional[float] = None,
                 early_stopping_enabled: bool = DEFAULT_EARLY_STOPPING_ENABLED) -> Tuple:
    """
    Training function with MLflow integration and Early Stopping.
    
    Args:
        experiment_name: Name of MLflow experiment
        run_name: Name of specific run (None = auto-generated with timestamp)
        batch_size: Batch size
        epochs: Maximum number of epochs
        embedding_dim: Embedding vector dimension
        margin: Margin for Triplet Loss
        samples_per_epoch: Number of triplets per epoch
        lr: Learning rate
        dataset_file: Path to dataset .pt file (None = use DEFAULT_DATASET_FILE)
        models_dir: Directory for saving models (None = use DEFAULT_MODELS_DIR)
        mlflow_tracking_uri: MLflow tracking URI (None = use DEFAULT_MLFLOW_TRACKING_URI)
        skip_types: Machine types to skip (e.g., ['fan'] to exclude fan data)
        early_stopping_patience: Number of epochs without improvement before stopping
        early_stopping_min_delta: Minimum improvement to consider epoch as improvement
        early_stopping_min_loss: Minimum loss threshold to stop training (None = disabled)
        early_stopping_enabled: Enable/disable early stopping
    
    Returns:
        tuple: (dataset, history, timestamp, mlflow_model_path)
    """
    # Use defaults from environment if not provided
    if dataset_file is None:
        dataset_file = DEFAULT_DATASET_FILE
    if models_dir is None:
        models_dir = DEFAULT_MODELS_DIR
    if mlflow_tracking_uri is None:
        mlflow_tracking_uri = DEFAULT_MLFLOW_TRACKING_URI
    if skip_types is None:
        skip_types = []
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
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
        if torch.backends.mps.is_available():
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
            "optimizer": "Adam",
            "loss_function": "TripletMarginLoss",
            "scheduler": "StepLR",
            "scheduler_step_size": 10,
            "scheduler_gamma": 0.1,
            "device": device_name,
            "dataset_file": dataset_file,
            "skip_types": f"{skip_types}",
            # Early Stopping parameters
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopping_min_loss": early_stopping_min_loss if early_stopping_min_loss else "None",
        })
        
        # Log tags
        mlflow.set_tags({
            "model_type": "TinyAudioCNN",
            "task": "triplet_learning",
            "dataset": "MIMII",
        })
        
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
        model = TinyAudioCNN(embedding_dim=embedding_dim).to(device)
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_params({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        })
        logger.info(f"📊 Model has {total_params:,} parameters ({trainable_params:,} trainable)")
        
        # 4. Loss function and Optimizer
        criterion = nn.TripletMarginLoss(margin=margin, p=2)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        logger.info(f"🏁 Starting training for {epochs} epochs...")
        
        # Initialize Early Stopping
        early_stopping = None
        if early_stopping_enabled:
            early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
                min_loss=early_stopping_min_loss if early_stopping_min_loss is not None else DEFAULT_EARLY_STOPPING_MIN_LOSS,
                mode='min',
                verbose=True,
                restore_best_weights=True
            )
            logger.info(f"🛡️ Early Stopping activated: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
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
            
            for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)
                
                optimizer.zero_grad()
                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)
                loss = criterion(emb_a, emb_p, emb_n)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
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
                       f"Loss: {avg_loss:.4f} | "
                       f"LR: {current_lr:.1e} | "
                       f"Time: {duration:.1f}s")
            
            # Check Early Stopping (BEFORE scheduler.step and model saving)
            if early_stopping is not None:
                if early_stopping(avg_loss, epoch, model):
                    stopped_early = True
                    logger.warning(f"\n🛑 Training stopped early at epoch {epoch+1}/{epochs}")
                    # Restore best weights
                    early_stopping.restore_weights(model)
                    # Update best_model_path with best weights
                    best_model_path = os.path.join(models_dir, f"{timestamp}_best_model.pth")
                    torch.save(model.state_dict(), best_model_path)
                    
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
                best_model_path = os.path.join(models_dir, f"{timestamp}_best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                mlflow.log_metric("best_loss", best_loss, step=epoch)
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(models_dir, f"{timestamp}_checkpoint_epoch_{epoch+1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path, "checkpoints")
        
        # Final training information
        if stopped_early:
            logger.warning("\n🛑 Training stopped early (Early Stopping)")
            if early_stopping is not None:
                es_info = early_stopping.get_info()
                logger.info(f"   Best result: {es_info['best_score']:.6f} at epoch {es_info['best_epoch']+1}")
                logger.info(f"   Last result: {history[-1]:.6f} at epoch {len(history)}")
        else:
            logger.info("\n🎉 Training completed!")
        
        logger.info(f"Minimum Loss: {best_loss:.4f}")
        logger.info(f"Best model: {best_model_path}")
        logger.info(f"Completed epochs: {len(history)}/{epochs}")
        
        # Save final version
        final_model_path = os.path.join(models_dir, f"{timestamp}_final_model.pth")
        torch.save(model.state_dict(), final_model_path)
        
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
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.plot(history)
            plt.title("Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Triplet Loss")
            plt.grid(True)
            if stopped_early and early_stopping is not None:
                es_info = early_stopping.get_info()
                plt.axvline(es_info['best_epoch'], color='green', linestyle='--',
                           label=f"Best epoch ({es_info['best_epoch']+1})")
                plt.axvline(len(history)-1, color='red', linestyle='--',
                           label=f"Stopped ({len(history)})")
            plt.legend()
            loss_plot_path = os.path.join(models_dir, f"{timestamp}_training_loss.png")
            plt.savefig(loss_plot_path)
            plt.close()
            mlflow.log_artifact(loss_plot_path, "plots")
            logger.info(f"Plot saved as {loss_plot_path}")
        except ImportError:
            logger.warning("matplotlib not available. Skipping plot generation.")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
        
        # Save history as artifact
        history_json_path = os.path.join(models_dir, f"{timestamp}_history.json")
        with open(history_json_path, 'w') as f:
            json.dump({"loss_history": history, "stopped_early": stopped_early}, f)
        mlflow.log_artifact(history_json_path, "history")
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"\n📊 MLflow run completed: {mlflow.get_tracking_uri()}")
        logger.info(f"Run ID: {run_id}")
        
        # Form MLflow path to best model
        if best_model_path and os.path.exists(best_model_path):
            mlflow_model_path = f"runs:/{run_id}/models/{os.path.basename(best_model_path)}"
            logger.info(f"📦 MLflow model path: {mlflow_model_path}")
        else:
            mlflow_model_path = None
            logger.warning("⚠️ Best model not found")
    
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
        '--models-dir', '-m',
        type=str,
        default=None,
        help=f'Directory for saving models (default: {DEFAULT_MODELS_DIR})'
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
        '--margin',
        type=float,
        default=DEFAULT_MARGIN,
        help=f'Triplet loss margin (default: {DEFAULT_MARGIN})'
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
        help=f'Learning rate (default: {DEFAULT_LEARNING_RATE})'
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
        '--no-early-stopping',
        action='store_true',
        help='Disable early stopping'
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
            dataset_file=args.dataset,
            models_dir=args.models_dir,
            mlflow_tracking_uri=args.mlflow_uri,
            skip_types=args.skip_types,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            early_stopping_min_loss=args.early_stopping_min_loss,
            early_stopping_enabled=not args.no_early_stopping
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
