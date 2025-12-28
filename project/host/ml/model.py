#!/usr/bin/env python3
"""
Tiny Audio CNN Model for Audio Anomaly Detection

This module provides a lightweight CNN model for audio embedding generation.
The model is designed for triplet loss training and produces normalized embeddings.

It can be run as a standalone script for testing or imported as a module for training.

Configuration:
    The module supports configuration via environment variables or .env file.
    See env.example for available configuration options.
    If python-dotenv is installed, .env file will be automatically loaded.

Usage as module:
    from model import TinyAudioCNN
    model = TinyAudioCNN(embedding_dim=64)
    embeddings = model(spectrogram_batch)

Usage as script:
    python model.py [--embedding-dim 64] [--test]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import logging

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
DEFAULT_EMBEDDING_DIM = int(os.getenv('MODEL_EMBEDDING_DIM', '64'))


class TinyAudioCNN(nn.Module):
    """
    Lightweight CNN model for audio embedding generation.
    
    The model processes mel spectrograms and produces normalized embeddings
    suitable for triplet loss training. It uses a series of convolutional blocks
    with batch normalization and max pooling, followed by global average pooling
    and a fully connected layer.
    
    Input shape: [Batch, 1, 64 (mel bins), 32 (time frames)]
    Output shape: [Batch, embedding_dim] (normalized to unit length)
    
    Architecture:
        - 4 convolutional blocks with increasing channels (16 -> 32 -> 64 -> 128)
        - Each block: Conv2d -> BatchNorm -> ReLU -> MaxPool2d
        - Global Average Pooling for variable input length
        - Fully connected layer to embedding dimension
        - L2 normalization (critical for triplet loss)
    
    Args:
        embedding_dim: Dimension of the output embedding vector (default: 64)
    """
    
    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super(TinyAudioCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input: [Batch, 1, 64 (mel), 32 (time)]
        # This shape comes from 1 second of audio with hop_length=512
        
        # --- Block 1 ---
        # Convolution: 1 channel (grayscale) -> 16 channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduce: 64x32 -> 32x16
        
        # --- Block 2 ---
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduce: 32x16 -> 16x8
        
        # --- Block 3 ---
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # Reduce: 16x8 -> 8x4
        
        # --- Block 4 (Final) ---
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        
        # Global Average Pooling:
        # Converts tensor [Batch, 128, 8, 4] -> [Batch, 128, 1, 1]
        # This makes the model independent of exact input duration
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Embedding Head ---
        # Compress 128 features into final vector of dimension embedding_dim
        self.fc = nn.Linear(128, embedding_dim)
        
        logger.debug(f"Initialized TinyAudioCNN with embedding_dim={embedding_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [Batch, 1, mel_bins, time_frames]
        
        Returns:
            Normalized embedding tensor of shape [Batch, embedding_dim]
        """
        # Pass through layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.flatten(1)  # [Batch, 128]
        
        x = self.fc(x)  # [Batch, embedding_dim]
        
        # --- NORMALIZATION (Critical for Triplet Loss) ---
        # All vectors become unit length (L2 norm = 1)
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
    def count_parameters(self) -> dict:
        """
        Count the number of trainable and total parameters in the model.
        
        Returns:
            Dictionary with 'total', 'trainable', and 'non_trainable' parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': non_trainable_params
        }
    
    def get_model_info(self) -> dict:
        """
        Get detailed information about the model architecture.
        
        Returns:
            Dictionary with model information including parameter counts and architecture details
        """
        params = self.count_parameters()
        return {
            'model_name': 'TinyAudioCNN',
            'embedding_dim': self.embedding_dim,
            'total_parameters': params['total'],
            'trainable_parameters': params['trainable'],
            'non_trainable_parameters': params['non_trainable'],
            'input_shape': '[Batch, 1, 64, 32]',
            'output_shape': f'[Batch, {self.embedding_dim}]',
            'normalization': 'L2 normalized (unit length vectors)'
        }


def test_model(embedding_dim: int = DEFAULT_EMBEDDING_DIM, batch_size: int = 2):
    """
    Test the model with dummy input and verify output properties.
    
    Args:
        embedding_dim: Dimension of embedding vector
        batch_size: Batch size for test input
    
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing TinyAudioCNN model...")
    
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
        if output.shape != expected_shape:
            logger.error(f"Output shape mismatch! Expected {expected_shape}, got {output.shape}")
            return False
        
        # 2. Check normalization (L2 norm should be ~1.0 for each vector)
        norms = torch.norm(output, p=2, dim=1)
        avg_norm = norms.mean().item()
        min_norm = norms.min().item()
        max_norm = norms.max().item()
        
        logger.info(f"Vector norms - Mean: {avg_norm:.6f}, Min: {min_norm:.6f}, Max: {max_norm:.6f}")
        
        # Allow small tolerance for floating point errors
        tolerance = 1e-5
        if abs(avg_norm - 1.0) > tolerance:
            logger.warning(f"Average norm ({avg_norm:.6f}) deviates from 1.0 by more than {tolerance}")
            return False
        
        # 3. Print model statistics
        params = model.count_parameters()
        logger.info("Model parameters:")
        logger.info(f"  Total: {params['total']:,}")
        logger.info(f"  Trainable: {params['trainable']:,}")
        logger.info(f"  Non-trainable: {params['non_trainable']:,}")
        
        # 4. Print model info
        model_info = model.get_model_info()
        logger.info("Model information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✅ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Test TinyAudioCNN model'
    )
    parser.add_argument(
        '--embedding-dim', '-e',
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
        help=f'Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=2,
        help='Batch size for test input (default: 2)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run model tests'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Print model information'
    )
    
    args = parser.parse_args()
    
    try:
        if args.info:
            model = TinyAudioCNN(embedding_dim=args.embedding_dim)
            model_info = model.get_model_info()
            logger.info("Model Information:")
            logger.info("=" * 60)
            for key, value in model_info.items():
                logger.info(f"{key:25}: {value}")
            logger.info("=" * 60)
            return 0
        
        if args.test:
            success = test_model(args.embedding_dim, args.batch_size)
            return 0 if success else 1
        
        # Default: run test
        logger.info("Running default test...")
        success = test_model(args.embedding_dim, args.batch_size)
        return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
