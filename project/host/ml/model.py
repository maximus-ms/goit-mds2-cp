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



class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution block.
    
    This is more efficient than standard convolution for mobile/embedded devices.
    It splits convolution into two steps:
    1. Depthwise: Separate spatial convolution for each input channel
    2. Pointwise: 1x1 convolution to mix channels
    
    For 3x3 kernel, this reduces parameters by ~8-9x compared to standard Conv2d.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of convolution kernel (default: 3)
        stride: Stride of convolution (default: 1)
        padding: Padding (default: 1)
        bias: Whether to use bias (default: False)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        
        # Depthwise convolution: separate filter for each input channel
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,  # Each channel processed separately
            bias=False
        )
        self.bn_depthwise = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution: 1x1 convolution to mix channels
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=bias
        )
        self.bn_pointwise = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = F.relu(x)
        
        x = self.pointwise(x)
        x = self.bn_pointwise(x)
        x = F.relu(x)
        
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual block (MobileNetV2 style).
    
    This block uses an inverted residual structure:
    1. Pointwise expansion (1x1 conv) - expands channels
    2. Depthwise convolution (3x3) - spatial processing
    3. Pointwise projection (1x1 conv) - compresses channels
    
    Uses ReLU6 activation and supports residual connections when input/output
    dimensions match and stride=1.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride of depthwise convolution (default: 1)
        expansion: Expansion ratio for hidden channels (default: 6)
    """
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(InvertedResidual, self).__init__()
        
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        # Expanded channels
        expanded_channels = in_channels * expansion
        
        # Pointwise expansion: 1x1 conv to expand channels
        self.expand = nn.Conv2d(in_channels, expanded_channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm2d(expanded_channels)
        
        # Depthwise convolution: 3x3 conv with groups=expanded_channels
        padding = 1 if stride == 1 else 0
        self.depthwise = nn.Conv2d(
            expanded_channels,
            expanded_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            groups=expanded_channels,
            bias=False
        )
        self.bn_depthwise = nn.BatchNorm2d(expanded_channels)
        
        # Pointwise projection: 1x1 conv to compress channels
        self.project = nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False)
        self.bn_project = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.expand(x)
        x = self.bn_expand(x)
        x = F.relu6(x)
        
        # Depthwise
        x = self.depthwise(x)
        x = self.bn_depthwise(x)
        x = F.relu6(x)
        
        # Projection (no activation here - linear bottleneck)
        x = self.project(x)
        x = self.bn_project(x)
        
        # Residual connection
        if self.use_residual:
            x = x + identity
        
        return x


class TinyAudioCNN(nn.Module):
    """
    Lightweight CNN model for audio embedding generation.
    
    The model processes mel spectrograms and produces normalized embeddings
    suitable for triplet loss training. It uses a series of standard convolutional blocks
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
            'architecture': 'Standard Convolutions (original)',
            'embedding_dim': self.embedding_dim,
            'total_parameters': params['total'],
            'trainable_parameters': params['trainable'],
            'non_trainable_parameters': params['non_trainable'],
            'input_shape': '[Batch, 1, 64, 32]',
            'output_shape': f'[Batch, {self.embedding_dim}]',
            'normalization': 'L2 normalized (unit length vectors)'
        }


class TinyAudioCNN_v2(nn.Module):
    """
    Lightweight CNN model for audio embedding generation (Optimized version for ESP32-S3).
    
    This is an optimized version of TinyAudioCNN that uses Depthwise Separable Convolutions
    to reduce the number of parameters and computational complexity while maintaining
    similar accuracy. Designed specifically for deployment on ESP32-S3.
    
    The model processes mel spectrograms and produces normalized embeddings
    suitable for triplet loss training.
    
    Input shape: [Batch, 1, 64 (mel bins), 32 (time frames)]
    Output shape: [Batch, embedding_dim] (normalized to unit length)
    
    Architecture:
        - First block: Standard Conv2d (1 -> 16 channels)
        - Blocks 2-4: Depthwise Separable Convolutions (16 -> 32 -> 64 -> 128)
        - Each block: Conv -> BatchNorm -> ReLU -> MaxPool2d
        - Global Average Pooling for variable input length
        - Fully connected layer to embedding dimension
        - L2 normalization (critical for triplet loss)
        
    Optimized for ESP32-S3:
        - Uses Depthwise Separable Convolutions to reduce parameters by ~5x
        - Reduces computational complexity (MACs) by ~8-9x
        - Maintains similar accuracy with much lower computational cost
        - Compatible with ESP-DL optimized kernels
    
    Args:
        embedding_dim: Dimension of the output embedding vector (default: 64)
    """
    
    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super(TinyAudioCNN_v2, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input: [Batch, 1, 64 (mel), 32 (time)]
        # This shape comes from 1 second of audio with hop_length=512
        
        # --- Block 1 ---
        # Standard convolution for first layer (1 channel -> 16 channels)
        # Depthwise separable doesn't help much here since input has only 1 channel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # Reduce: 64x32 -> 32x16
        
        # --- Block 2 ---
        # Depthwise Separable Convolution: 16 -> 32 channels
        # More efficient than standard Conv2d for ESP32-S3
        self.conv2 = DepthwiseSeparableConv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Reduce: 32x16 -> 16x8
        
        # --- Block 3 ---
        # Depthwise Separable Convolution: 32 -> 64 channels
        self.conv3 = DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # Reduce: 16x8 -> 8x4
        
        # --- Block 4 (Final) ---
        # Depthwise Separable Convolution: 64 -> 128 channels
        self.conv4 = DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1)
        
        # Global Average Pooling:
        # Converts tensor [Batch, 128, 8, 4] -> [Batch, 128, 1, 1]
        # This makes the model independent of exact input duration
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Embedding Head ---
        # Compress 128 features into final vector of dimension embedding_dim
        self.fc = nn.Linear(128, embedding_dim)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [Batch, 1, mel_bins, time_frames]
        
        Returns:
            Normalized embedding tensor of shape [Batch, embedding_dim]
        """
        # Pass through layers
        # Block 1: Standard convolution
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # Blocks 2-4: Depthwise Separable Convolutions (already include ReLU)
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv4(x)  # Final block, no pooling
        
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
            'model_name': 'TinyAudioCNN_v2',
            'architecture': 'Depthwise Separable Convolutions (optimized for ESP32-S3)',
            'embedding_dim': self.embedding_dim,
            'total_parameters': params['total'],
            'trainable_parameters': params['trainable'],
            'non_trainable_parameters': params['non_trainable'],
            'input_shape': '[Batch, 1, 64, 32]',
            'output_shape': f'[Batch, {self.embedding_dim}]',
            'normalization': 'L2 normalized (unit length vectors)'
        }


class TinyAudioCNN_v3(nn.Module):
    """
    Lightweight CNN model for audio embedding generation (MobileNetV2 style with Inverted Residuals).
    
    This version uses Inverted Residual blocks (MobileNetV2 style) to achieve better
    capacity (~50-60k parameters) while maintaining efficiency for edge devices.
    The model is designed to balance between model capacity and computational efficiency.
    
    The model processes mel spectrograms and produces normalized embeddings
    suitable for triplet loss training.
    
    Input shape: [Batch, 1, 64 (mel bins), 32 (time frames)]
    Output shape: [Batch, embedding_dim] (normalized to unit length)
    
    Architecture:
        - First block: Standard Conv2d (1 -> 24 channels)
        - Multiple Inverted Residual blocks with expansion ratio 2
        - Blocks use MobileNetV2 style: expand -> depthwise -> project
        - Global Average Pooling for variable input length
        - Fully connected layer to embedding dimension
        - L2 normalization (critical for triplet loss)
        
    Optimized for edge devices:
        - Uses Inverted Residual blocks for efficient feature extraction
        - ~50-60k parameters (balanced capacity vs efficiency)
        - Better feature representation than v2 while remaining efficient
        - Compatible with mobile/embedded deployment
    
    Args:
        embedding_dim: Dimension of the output embedding vector (default: 64)
    """
    
    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super(TinyAudioCNN_v3, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # Input: [Batch, 1, 64 (mel), 32 (time)]
        # This shape comes from 1 second of audio with hop_length=512
        
        # --- Initial Block ---
        # Standard convolution for first layer (1 channel -> 16 channels)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        # After conv1: 64x32 -> 32x16
        
        # --- Inverted Residual Blocks ---
        # Block 1: 16 -> 24 channels, stride=1 (no downsampling)
        self.block1 = InvertedResidual(16, 24, stride=1, expansion=2)
        
        # Block 2: 24 -> 24 channels, stride=2 (downsampling)
        self.block2 = InvertedResidual(24, 24, stride=2, expansion=2)
        # After block2: 32x16 -> 16x8
        
        # Block 3: 24 -> 32 channels, stride=1
        self.block3 = InvertedResidual(24, 32, stride=1, expansion=2)
        
        # Block 4: 32 -> 32 channels, stride=2 (downsampling)
        self.block4 = InvertedResidual(32, 32, stride=2, expansion=2)
        # After block4: 16x8 -> 8x4
        
        # Block 5: 32 -> 48 channels, stride=1
        self.block5 = InvertedResidual(32, 48, stride=1, expansion=2)
        
        # Block 6: 48 -> 64 channels, stride=1
        self.block6 = InvertedResidual(48, 64, stride=1, expansion=2)
        
        # Block 7: 64 -> 80 channels, stride=1
        self.block7 = InvertedResidual(64, 80, stride=1, expansion=2)
        
        # Global Average Pooling:
        # Converts tensor [Batch, 128, 8, 4] -> [Batch, 128, 1, 1]
        # This makes the model independent of exact input duration
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # --- Embedding Head ---
        # Compress 80 features into final vector of dimension embedding_dim
        self.fc = nn.Linear(80, embedding_dim)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [Batch, 1, mel_bins, time_frames]
        
        Returns:
            Normalized embedding tensor of shape [Batch, embedding_dim]
        """
        # Initial block
        x = F.relu6(self.bn1(self.conv1(x)))  # [B, 24, 32, 16]
        
        # Inverted Residual blocks
        x = self.block1(x)  # [B, 24, 32, 16]
        x = self.block2(x)  # [B, 24, 16, 8]
        x = self.block3(x)  # [B, 32, 16, 8]
        x = self.block4(x)  # [B, 32, 8, 4]
        x = self.block5(x)  # [B, 48, 8, 4]
        x = self.block6(x)  # [B, 64, 8, 4]
        x = self.block7(x)  # [B, 80, 8, 4]
        
        # Global pooling and embedding
        x = self.global_pool(x)  # [B, 80, 1, 1]
        x = x.flatten(1)  # [B, 80]
        
        x = self.fc(x)  # [B, embedding_dim]
        
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
            'model_name': 'TinyAudioCNN_v3',
            'architecture': 'Inverted Residuals (MobileNetV2 style)',
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
        
        # Initialize model (test all versions)
        model_v1 = TinyAudioCNN(embedding_dim=embedding_dim)
        model_v2 = TinyAudioCNN_v2(embedding_dim=embedding_dim)
        model_v3 = TinyAudioCNN_v3(embedding_dim=embedding_dim)
        logger.info(f"Models initialized with embedding_dim={embedding_dim}")
        
        # Test v1
        logger.info("\n--- Testing TinyAudioCNN (original) ---")
        model = model_v1
        
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
        
        # Test v2
        logger.info("\n--- Testing TinyAudioCNN_v2 (optimized) ---")
        model = model_v2
        model.eval()
        with torch.no_grad():
            output_v2 = model(dummy_input)
        
        # Verify v2 output properties
        if output_v2.shape != expected_shape:
            logger.error(f"V2 Output shape mismatch! Expected {expected_shape}, got {output_v2.shape}")
            return False
        
        norms_v2 = torch.norm(output_v2, p=2, dim=1)
        avg_norm_v2 = norms_v2.mean().item()
        if abs(avg_norm_v2 - 1.0) > tolerance:
            logger.warning(f"V2 Average norm ({avg_norm_v2:.6f}) deviates from 1.0")
            return False
        
        params_v2 = model.count_parameters()
        logger.info("V2 Model parameters:")
        logger.info(f"  Total: {params_v2['total']:,}")
        logger.info(f"  Trainable: {params_v2['trainable']:,}")
        
        model_info_v2 = model.get_model_info()
        logger.info("V2 Model information:")
        for key, value in model_info_v2.items():
            logger.info(f"  {key}: {value}")
        
        # Test v3
        logger.info("\n--- Testing TinyAudioCNN_v3 (MobileNetV2 style) ---")
        model = model_v3
        model.eval()
        with torch.no_grad():
            output_v3 = model(dummy_input)
        
        # Verify v3 output properties
        if output_v3.shape != expected_shape:
            logger.error(f"V3 Output shape mismatch! Expected {expected_shape}, got {output_v3.shape}")
            return False
        
        norms_v3 = torch.norm(output_v3, p=2, dim=1)
        avg_norm_v3 = norms_v3.mean().item()
        if abs(avg_norm_v3 - 1.0) > tolerance:
            logger.warning(f"V3 Average norm ({avg_norm_v3:.6f}) deviates from 1.0")
            return False
        
        params_v3 = model.count_parameters()
        logger.info("V3 Model parameters:")
        logger.info(f"  Total: {params_v3['total']:,}")
        logger.info(f"  Trainable: {params_v3['trainable']:,}")
        
        model_info_v3 = model.get_model_info()
        logger.info("V3 Model information:")
        for key, value in model_info_v3.items():
            logger.info(f"  {key}: {value}")
        
        # Comparison
        logger.info("\n--- Comparison ---")
        params_v1 = model_v1.count_parameters()
        logger.info(f"V1 Parameters: {params_v1['total']:,}")
        logger.info(f"V2 Parameters: {params_v2['total']:,}")
        logger.info(f"V3 Parameters: {params_v3['total']:,}")
        logger.info(f"V1->V2 Reduction: {params_v1['total'] / params_v2['total']:.2f}x")
        logger.info(f"V2->V3 Increase: {params_v3['total'] / params_v2['total']:.2f}x")
        logger.info(f"V3 vs V1: {params_v3['total'] / params_v1['total']:.2f}x")
        
        logger.info("\n✅ All tests passed!")
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
            logger.info("=" * 70)
            logger.info("TinyAudioCNN (Original)")
            logger.info("=" * 70)
            model_v1 = TinyAudioCNN(embedding_dim=args.embedding_dim)
            model_info_v1 = model_v1.get_model_info()
            for key, value in model_info_v1.items():
                logger.info(f"{key:25}: {value}")
            
            logger.info("\n" + "=" * 70)
            logger.info("TinyAudioCNN_v2 (Optimized for ESP32-S3)")
            logger.info("=" * 70)
            model_v2 = TinyAudioCNN_v2(embedding_dim=args.embedding_dim)
            model_info_v2 = model_v2.get_model_info()
            for key, value in model_info_v2.items():
                logger.info(f"{key:25}: {value}")
            
            logger.info("\n" + "=" * 70)
            logger.info("TinyAudioCNN_v3 (MobileNetV2 style)")
            logger.info("=" * 70)
            model_v3 = TinyAudioCNN_v3(embedding_dim=args.embedding_dim)
            model_info_v3 = model_v3.get_model_info()
            for key, value in model_info_v3.items():
                logger.info(f"{key:25}: {value}")
            
            logger.info("\n" + "=" * 70)
            logger.info("Comparison")
            logger.info("=" * 70)
            logger.info(f"V1 Parameters: {model_info_v1['total_parameters']:,}")
            logger.info(f"V2 Parameters: {model_info_v2['total_parameters']:,}")
            logger.info(f"V3 Parameters: {model_info_v3['total_parameters']:,}")
            reduction_v1_v2 = model_info_v1['total_parameters'] / model_info_v2['total_parameters']
            increase_v2_v3 = model_info_v3['total_parameters'] / model_info_v2['total_parameters']
            ratio_v3_v1 = model_info_v3['total_parameters'] / model_info_v1['total_parameters']
            logger.info(f"V1->V2 Reduction: {reduction_v1_v2:.2f}x")
            logger.info(f"V2->V3 Increase: {increase_v2_v3:.2f}x")
            logger.info(f"V3 vs V1: {ratio_v3_v1:.2f}x")
            logger.info("=" * 70)
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
