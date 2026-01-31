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


def test_tiny_model():
    print("test_tiny_model")


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
        
        logger.debug(f"Initialized TinyAudioCNN_v2 (optimized) with embedding_dim={embedding_dim}")
    
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
        
        logger.debug(f"Initialized TinyAudioCNN_v3 (MobileNetV2 style) with embedding_dim={embedding_dim}")
    
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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block: 'Attention' for channels."""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        # Squeeze: Global Average Pooling (вже вбудовано в логіку forward)
        # Excitation: 2 маленьких FC шари
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Global Average Pooling: [B, C, H, W] -> [B, C]
        y = x.mean(dim=(2, 3))
        # Обчислюємо вагу кожного каналу
        y = self.fc(y).view(b, c, 1, 1)
        # Множимо вхід на ваги (масштабуємо канали)
        return x * y

class MBConvBlock(nn.Module):
    """
    MobileNetV2 style Inverted Residual Block with SE-Attention.
    Структура: Expand(1x1) -> Depthwise(3x3) -> SE -> Project(1x1)
    """
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super().__init__()
        self.use_residual = (in_channels == out_channels) and (stride == 1)
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        # 1. Expansion Phase (1x1 Conv)
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True)) # ReLU6 краще для квантування на ESP32
        
        # 2. Depthwise Convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, 
                                stride=stride, padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # 3. Squeeze-and-Excitation (Attention)
        layers.append(SEBlock(hidden_dim))
        
        # 4. Projection Phase (1x1 Conv) - лінійний вихід (без ReLU)
        layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x) # Skip connection
        else:
            return self.conv(x)

class TinyAudioCNN_v4(nn.Module):
    """
    Lightweight CNN model for audio embedding generation (EfficientNet-like with SE-Attention).
    
    This version uses MBConv blocks (MobileNetV2 style Inverted Residuals) with 
    Squeeze-and-Excitation attention for improved feature extraction. The model
    is designed for deployment on ESP32-S3 with good balance of accuracy and efficiency.
    
    The model processes mel spectrograms and produces normalized embeddings
    suitable for triplet loss training.
    
    Input shape: [Batch, 1, 64 (mel bins), 32 (time frames)]
    Output shape: [Batch, embedding_dim] (normalized to unit length)
    
    Architecture:
        - Stem: Standard Conv2d (1 -> 16 channels, stride=2)
        - 6 MBConv blocks with SE-Attention
        - Expansion ratios: 1, 4, 4, 4, 4, 4
        - Channel progression: 16 -> 16 -> 32 -> 32 -> 64 -> 64 -> 128
        - Global Average Pooling for variable input length
        - Fully connected layer to embedding dimension
        - L2 normalization (critical for triplet loss)
        
    Optimized for edge devices:
        - Uses MBConv blocks with SE-Attention for better feature learning
        - ReLU6 activations for better quantization on ESP32
        - Efficient channel expansion with depthwise separable convolutions
        - Skip connections for better gradient flow
    
    Args:
        embedding_dim: Dimension of the output embedding vector (default: 64)
    """
    
    def __init__(self, embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # --- STEM ---
        # Перший шар залишаємо простим, щоб не забити RAM
        # [Batch, 1, 64, 32] -> [Batch, 16, 32, 16] (stride 2 зменшує розмір)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        
        # --- BACKBONE (MobileNetV2 blocks) ---
        # t = expand_ratio (наскільки розширювати канали всередині блоку)
        # c = out_channels
        # s = stride
        self.blocks = nn.Sequential(
            # Block 1: 16 -> 16 (Refining, no downsample)
            MBConvBlock(in_channels=16, out_channels=16, expand_ratio=1, stride=1),
            
            # Block 2: 16 -> 32 (Downsample: 32x16 -> 16x8)
            MBConvBlock(in_channels=16, out_channels=32, expand_ratio=4, stride=2),
            # Block 3: 32 -> 32 (Refining features + Residual)
            MBConvBlock(in_channels=32, out_channels=32, expand_ratio=4, stride=1),
            
            # Block 4: 32 -> 64 (Downsample: 16x8 -> 8x4)
            MBConvBlock(in_channels=32, out_channels=64, expand_ratio=4, stride=2),
            # Block 5: 64 -> 64 (Deepening + Residual)
            MBConvBlock(in_channels=64, out_channels=64, expand_ratio=4, stride=1),
            
            # Block 6: 64 -> 128 (Final features)
            MBConvBlock(in_channels=64, out_channels=128, expand_ratio=4, stride=1)
        )
        
        # --- HEAD ---
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embedding_dim)
        
        logger.debug(f"Initialized TinyAudioCNN_v4 (EfficientNet-like with SE) with embedding_dim={embedding_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [Batch, 1, mel_bins, time_frames]
        
        Returns:
            Normalized embedding tensor of shape [Batch, embedding_dim]
        """
        x = self.stem(x)
        x = self.blocks(x)
        
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        
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
            'model_name': 'TinyAudioCNN_v4',
            'architecture': 'MBConv + SE-Attention (EfficientNet-like)',
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
    logger.info("Testing TinyAudioCNN models (v1-v4)...")
    
    try:
        # Create dummy input: [Batch, 1, 64 (mel), 32 (time)]
        # This simulates 1 second of audio with hop_length=512
        dummy_input = torch.randn(batch_size, 1, 64, 32)
        logger.info(f"Input shape: {dummy_input.shape}")
        
        # Initialize all model versions
        models = {
            'v1': TinyAudioCNN(embedding_dim=embedding_dim),
            'v2': TinyAudioCNN_v2(embedding_dim=embedding_dim),
            'v3': TinyAudioCNN_v3(embedding_dim=embedding_dim),
            'v4': TinyAudioCNN_v4(embedding_dim=embedding_dim),
        }
        logger.info(f"All models initialized with embedding_dim={embedding_dim}")
        
        expected_shape = (batch_size, embedding_dim)
        tolerance = 1e-5
        
        # Test each model version
        for version, model in models.items():
            model_info = model.get_model_info()
            logger.info(f"\n--- Testing {model_info['model_name']} ({model_info['architecture']}) ---")
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            logger.info(f"Output shape: {output.shape}")
            
            # Verify output shape
            if output.shape != expected_shape:
                logger.error(f"{version.upper()} Output shape mismatch! Expected {expected_shape}, got {output.shape}")
                return False
            
            # Verify normalization
            norms = torch.norm(output, p=2, dim=1)
            avg_norm = norms.mean().item()
            min_norm = norms.min().item()
            max_norm = norms.max().item()
            
            logger.info(f"Vector norms - Mean: {avg_norm:.6f}, Min: {min_norm:.6f}, Max: {max_norm:.6f}")
            
            if abs(avg_norm - 1.0) > tolerance:
                logger.warning(f"{version.upper()} Average norm ({avg_norm:.6f}) deviates from 1.0")
                return False
            
            # Print model statistics
            params = model.count_parameters()
            logger.info("Model parameters:")
            logger.info(f"  Total: {params['total']:,}")
            logger.info(f"  Trainable: {params['trainable']:,}")
            logger.info(f"  Non-trainable: {params['non_trainable']:,}")
        
        # Comparison
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON OF ALL MODEL VERSIONS")
        logger.info("=" * 70)
        
        params_all = {v: m.count_parameters()['total'] for v, m in models.items()}
        
        for version, model in models.items():
            info = model.get_model_info()
            logger.info(f"\n{info['model_name']}:")
            logger.info(f"  Architecture: {info['architecture']}")
            logger.info(f"  Parameters:   {params_all[version]:,}")
        
        logger.info("\n--- Parameter Ratios ---")
        logger.info(f"V1 (baseline):      {params_all['v1']:,} params")
        logger.info(f"V2 vs V1:           {params_all['v2'] / params_all['v1']:.3f}x ({params_all['v1'] / params_all['v2']:.2f}x reduction)")
        logger.info(f"V3 vs V1:           {params_all['v3'] / params_all['v1']:.3f}x")
        logger.info(f"V4 vs V1:           {params_all['v4'] / params_all['v1']:.3f}x")
        logger.info(f"V4 vs V3:           {params_all['v4'] / params_all['v3']:.3f}x")
        
        logger.info("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False


def compare_all_models(embedding_dim: int = DEFAULT_EMBEDDING_DIM, batch_size: int = 2):
    """
    Comprehensive comparison of all TinyAudioCNN model versions.
    
    Compares: parameter counts, layer counts, FLOPs estimation, memory usage.
    
    Args:
        embedding_dim: Dimension of embedding vector
        batch_size: Batch size for test input
    """
    import time
    
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE MODEL COMPARISON")
    logger.info("=" * 80)
    
    # Initialize all models
    models = {
        'TinyAudioCNN (v1)': TinyAudioCNN(embedding_dim=embedding_dim),
        'TinyAudioCNN_v2': TinyAudioCNN_v2(embedding_dim=embedding_dim),
        'TinyAudioCNN_v3': TinyAudioCNN_v3(embedding_dim=embedding_dim),
        'TinyAudioCNN_v4': TinyAudioCNN_v4(embedding_dim=embedding_dim),
    }
    
    dummy_input = torch.randn(batch_size, 1, 64, 32)
    
    results = []
    
    for name, model in models.items():
        model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count layers
        conv_layers = 0
        bn_layers = 0
        linear_layers = 0
        other_layers = 0
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                conv_layers += 1
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                bn_layers += 1
            elif isinstance(module, nn.Linear):
                linear_layers += 1
            elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.AdaptiveAvgPool2d, nn.MaxPool2d)):
                other_layers += 1
        
        # Measure inference time (average over multiple runs)
        num_runs = 100
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(dummy_input)
            
            # Timed runs
            start_time = time.perf_counter()
            for _ in range(num_runs):
                _ = model(dummy_input)
            end_time = time.perf_counter()
        
        avg_inference_ms = (end_time - start_time) / num_runs * 1000
        
        # Estimate model size in KB (float32 = 4 bytes per param)
        model_size_kb = total_params * 4 / 1024
        
        results.append({
            'name': name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'conv_layers': conv_layers,
            'bn_layers': bn_layers,
            'linear_layers': linear_layers,
            'other_layers': other_layers,
            'total_layers': conv_layers + bn_layers + linear_layers,
            'inference_ms': avg_inference_ms,
            'model_size_kb': model_size_kb,
        })
    
    # Print comparison table
    logger.info(f"\n{'Model':<25} {'Params':>12} {'Conv':>6} {'BN':>6} {'FC':>6} {'Layers':>8} {'Size(KB)':>10} {'Infer(ms)':>10}")
    logger.info("-" * 95)
    
    baseline_params = results[0]['total_params']
    
    for r in results:
        ratio = r['total_params'] / baseline_params
        logger.info(
            f"{r['name']:<25} {r['total_params']:>12,} {r['conv_layers']:>6} {r['bn_layers']:>6} "
            f"{r['linear_layers']:>6} {r['total_layers']:>8} {r['model_size_kb']:>10.1f} {r['inference_ms']:>10.3f}"
        )
    
    logger.info("-" * 95)
    logger.info("\n--- Relative to V1 (baseline) ---")
    for r in results:
        ratio = r['total_params'] / baseline_params
        logger.info(f"{r['name']:<25}: {ratio:.3f}x params ({r['total_params']:,})")
    
    logger.info("\n--- Model Descriptions ---")
    descriptions = {
        'TinyAudioCNN (v1)': 'Standard convolutions - baseline model',
        'TinyAudioCNN_v2': 'Depthwise separable convolutions - optimized for ESP32-S3',
        'TinyAudioCNN_v3': 'Inverted residuals (MobileNetV2 style) - balanced capacity',
        'TinyAudioCNN_v4': 'MBConv + SE-Attention (EfficientNet-like) - best features',
    }
    for name, desc in descriptions.items():
        logger.info(f"  {name}: {desc}")
    
    logger.info("\n" + "=" * 80)
    
    return results


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
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run comprehensive comparison of all model versions'
    )
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            compare_all_models(args.embedding_dim, args.batch_size)
            return 0
        
        if args.info:
            # All model versions
            model_classes = [
                ('TinyAudioCNN (Original)', TinyAudioCNN),
                ('TinyAudioCNN_v2 (Depthwise Separable)', TinyAudioCNN_v2),
                ('TinyAudioCNN_v3 (MobileNetV2 style)', TinyAudioCNN_v3),
                ('TinyAudioCNN_v4 (MBConv + SE-Attention)', TinyAudioCNN_v4),
            ]
            
            model_infos = []
            for title, model_class in model_classes:
                logger.info("=" * 70)
                logger.info(title)
                logger.info("=" * 70)
                model = model_class(embedding_dim=args.embedding_dim)
                model_info = model.get_model_info()
                model_infos.append(model_info)
                for key, value in model_info.items():
                    logger.info(f"{key:25}: {value}")
                logger.info("")
            
            # Comparison
            logger.info("=" * 70)
            logger.info("PARAMETER COMPARISON")
            logger.info("=" * 70)
            for info in model_infos:
                logger.info(f"{info['model_name']:20}: {info['total_parameters']:>10,} params")
            
            logger.info("\n--- Ratios relative to V1 ---")
            baseline = model_infos[0]['total_parameters']
            for info in model_infos:
                ratio = info['total_parameters'] / baseline
                logger.info(f"{info['model_name']:20}: {ratio:.3f}x")
            
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
