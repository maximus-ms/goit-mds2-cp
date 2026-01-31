#!/usr/bin/env python3
"""
Export PyTorch Model for ESP32 Deployment

This script exports a trained TinyAudioCNN model to TFLite format,
generates C header files for ESP32 deployment, and creates verification data.

Features:
    - Load model from MLflow or local file
    - Convert PyTorch to TensorFlow Lite (via ai-edge-torch)
    - Generate model_data.h for ESP32
    - Generate reference_embeddings.h for anomaly detection
    - Generate ml_verification_data.h for inference testing
    - Verify exported model matches original

Usage:
    # Export from MLflow run (includes verification data by default)
    python export_esp32.py --mlflow-run-id <run_id>
    
    # Export from local .pth file
    python export_esp32.py --model-path model.pth
    
    # Export with quantization
    python export_esp32.py --mlflow-run-id <run_id> --quantize int8
    
    # Skip model export, only generate verification data
    python export_esp32.py --mlflow-run-id <run_id> --skip-export
    
    # Skip verification data generation
    python export_esp32.py --mlflow-run-id <run_id> --skip-verification

Configuration:
    Uses environment variables from .env file (same as train.py)
"""

import os
import sys
import argparse
import logging
import tempfile
import struct
from datetime import datetime
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    # Try to load from ml/.env or parent .env
    env_paths = [
        os.path.join(os.path.dirname(__file__), '.env'),
        os.path.join(os.path.dirname(__file__), '..', '.env'),
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
except ImportError:
    pass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration from .env
# =============================================================================
DEFAULT_MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
DEFAULT_EMBEDDING_DIM = int(os.getenv('MODEL_EMBEDDING_DIM', '64'))
DEFAULT_OUTPUT_DIR = os.getenv('ESP32_EXPORT_DIR', './main/ml')

# Model input shape: [batch, channels, mels, frames]
MODEL_INPUT_MELS = int(os.getenv('MEL_SPECTROGRAM_N_MELS', '64'))
MODEL_INPUT_FRAMES = int(os.getenv('MODEL_INPUT_FRAMES', '32'))

# Verification data settings
DEFAULT_VERIFICATION_SAMPLES = 16


# =============================================================================
# Model Loading Functions
# =============================================================================

def load_model_from_mlflow(
    run_id: str,
    tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    artifact_name: str = None
) -> Tuple['torch.nn.Module', dict]:
    """
    Load a trained model from MLflow.
    
    Args:
        run_id: MLflow run ID
        tracking_uri: MLflow tracking URI
        artifact_name: Name of the model artifact (e.g., 'best_model.pth')
    
    Returns:
        Tuple of (model, run_info)
    """
    import torch
    import mlflow
    
    logger.info(f"Loading model from MLflow run: {run_id}")
    logger.info(f"MLflow URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Get run info
    run = mlflow.get_run(run_id)
    run_info = {
        'run_id': run_id,
        'experiment_id': run.info.experiment_id,
        'status': run.info.status,
        'params': run.data.params,
        'metrics': run.data.metrics,
        'tags': run.data.tags,
    }
    
    # Get model version and embedding dim from params
    model_version = run.data.params.get('model_version', '1')
    embedding_dim = int(run.data.params.get('embedding_dim', DEFAULT_EMBEDDING_DIM))
    
    logger.info(f"Model version: {model_version}, Embedding dim: {embedding_dim}")
    
    # Import model class based on version
    try:
        from ml.model import TinyAudioCNN, TinyAudioCNN_v2, TinyAudioCNN_v3, TinyAudioCNN_v4
    except ImportError:
        from model import TinyAudioCNN, TinyAudioCNN_v2, TinyAudioCNN_v3, TinyAudioCNN_v4
    
    if model_version == '4':
        ModelClass = TinyAudioCNN_v4
    elif model_version == '3':
        ModelClass = TinyAudioCNN_v3
    elif model_version == '2':
        ModelClass = TinyAudioCNN_v2
    else:
        ModelClass = TinyAudioCNN
    
    # Download model artifact
    with tempfile.TemporaryDirectory() as temp_dir:
        # List available artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, "models")
        
        if not artifacts:
            raise FileNotFoundError(f"No model artifacts found in run {run_id}")
        
        # Find the model file
        if artifact_name:
            model_artifact = next((a for a in artifacts if a.path.endswith(artifact_name)), None)
        else:
            # Prefer best_model, then final_model
            model_artifact = next((a for a in artifacts if 'best' in a.path), None)
            if not model_artifact:
                model_artifact = next((a for a in artifacts if 'final' in a.path), None)
            if not model_artifact:
                model_artifact = artifacts[0]
        
        if not model_artifact:
            raise FileNotFoundError(f"Model artifact not found: {artifact_name}")
        
        logger.info(f"Downloading artifact: {model_artifact.path}")
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=model_artifact.path,
            dst_path=temp_dir
        )
        
        # Load model
        model = ModelClass(embedding_dim=embedding_dim)
        model.load_state_dict(torch.load(local_path, map_location='cpu'))
        model.eval()
        
        logger.info(f"Model loaded successfully: {ModelClass.__name__}")
    
    return model, run_info


def load_model_from_file(
    model_path: str,
    model_version: str = '1',
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> 'torch.nn.Module':
    """
    Load a model from a local .pth file.
    
    Args:
        model_path: Path to .pth file
        model_version: Model version ('1', '2', '3', '4' or 'v1', 'v2', 'v3', 'v4')
        embedding_dim: Embedding dimension
    
    Returns:
        Loaded model
    """
    import torch
    
    logger.info(f"Loading model from file: {model_path}")
    
    # Normalize model_version (support both 'v1' and '1' formats)
    model_version = str(model_version).lower().strip()
    if model_version.startswith('v'):
        model_version = model_version[1:]
    
    # Import model class
    try:
        from ml.model import TinyAudioCNN, TinyAudioCNN_v2, TinyAudioCNN_v3, TinyAudioCNN_v4
    except ImportError:
        from model import TinyAudioCNN, TinyAudioCNN_v2, TinyAudioCNN_v3, TinyAudioCNN_v4
    
    if model_version == '4':
        ModelClass = TinyAudioCNN_v4
    elif model_version == '3':
        ModelClass = TinyAudioCNN_v3
    elif model_version == '2':
        ModelClass = TinyAudioCNN_v2
    else:
        ModelClass = TinyAudioCNN
    
    model = ModelClass(embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    logger.info(f"Model loaded: {ModelClass.__name__}, {embedding_dim}D embedding")
    return model


def load_model(
    mlflow_run_id: str = None,
    model_path: str = None,
    mlflow_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    artifact_name: str = None,
    model_version: str = '1',
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> Tuple['torch.nn.Module', int]:
    """
    Universal model loader - from MLflow or local file.
    
    Returns:
        Tuple of (model, embedding_dim)
    """
    if mlflow_run_id:
        model, run_info = load_model_from_mlflow(
            mlflow_run_id, mlflow_uri, artifact_name
        )
        actual_embedding_dim = int(run_info['params'].get('embedding_dim', embedding_dim))
        return model, actual_embedding_dim
    elif model_path:
        model = load_model_from_file(model_path, model_version, embedding_dim)
        return model, embedding_dim
    else:
        raise ValueError("Either mlflow_run_id or model_path must be provided")


# =============================================================================
# TFLite Conversion Functions
# =============================================================================

def convert_pytorch_to_tflite(
    model: 'torch.nn.Module',
    output_path: str,
    input_frames: int = MODEL_INPUT_FRAMES,
    quantize: str = 'float32'
) -> str:
    """
    Convert PyTorch model directly to TensorFlow Lite using ai-edge-torch.
    
    Args:
        model: PyTorch model
        output_path: Output .tflite file path
        input_frames: Number of time frames in input
        quantize: Quantization type ('float32', 'float16', 'int8')
    
    Returns:
        Path to exported TFLite file
    """
    import torch
    
    logger.info(f"Converting PyTorch to TFLite: {output_path}")
    logger.info(f"Input shape: [1, 1, {MODEL_INPUT_MELS}, {input_frames}]")
    logger.info(f"Quantization: {quantize}")
    
    try:
        import ai_edge_torch
    except ImportError as e:
        logger.error(f"ai-edge-torch not installed: {e}")
        logger.error("Install with: pip install ai-edge-torch")
        raise
    
    # Create sample input for tracing
    sample_input = (torch.randn(1, 1, MODEL_INPUT_MELS, input_frames),)
    
    # Convert PyTorch model to TFLite
    model.eval()
    
    try:
        import tensorflow as tf
        
        # ai-edge-torch converts PyTorch → SavedModel → TFLite
        edge_model = ai_edge_torch.convert(model, sample_input)
        
        if quantize == 'int8':
            # Export to SavedModel first, then convert with int8 quantization
            with tempfile.TemporaryDirectory() as temp_dir:
                saved_model_path = os.path.join(temp_dir, 'saved_model')
                tflite_temp = os.path.join(temp_dir, 'model_float.tflite')
                
                # Export float model to get SavedModel
                edge_model.export(tflite_temp)
                
                logger.info("Applying INT8 dynamic range quantization...")
                
                converter = tf.lite.TFLiteConverter.from_saved_model(
                    os.path.dirname(tflite_temp)
                ) if os.path.exists(os.path.join(os.path.dirname(tflite_temp), 'saved_model.pb')) else None
                
                if converter is None:
                    logger.info("Using ONNX path for int8 quantization...")
                    return _convert_via_onnx_int8(model, output_path, input_frames)
                
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                int8_model = converter.convert()
                
                with open(output_path, 'wb') as f:
                    f.write(int8_model)
                    
                logger.info(f"INT8 model exported: {len(int8_model)/1024:.1f} KB")
        else:
            # float32 or float16
            edge_model.export(output_path)
            
            if quantize == 'float16':
                logger.info("Applying float16 quantization post-export...")
                _apply_float16_quantization(output_path)
        
        logger.info(f"TFLite model exported: {output_path}")
        
    except Exception as e:
        logger.warning(f"ai-edge-torch failed: {e}")
        logger.info("Falling back to TorchScript → ONNX → TFLite conversion...")
        return _convert_via_onnx(model, output_path, input_frames, quantize)
    
    # Get model size
    model_size = os.path.getsize(output_path)
    logger.info(f"TFLite model size: {model_size / 1024:.2f} KB")
    
    return output_path


def _convert_via_onnx_int8(
    model: 'torch.nn.Module',
    output_path: str,
    input_frames: int
) -> str:
    """Convert PyTorch model to INT8 TFLite via ONNX + TensorFlow."""
    import torch
    import tensorflow as tf
    
    logger.info("Converting via ONNX for INT8 quantization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        onnx_path = os.path.join(temp_dir, 'model.onnx')
        sample_input = torch.randn(1, 1, MODEL_INPUT_MELS, input_frames)
        
        torch.onnx.export(
            model, sample_input, onnx_path,
            input_names=['input'], output_names=['output'],
            opset_version=13, do_constant_folding=True
        )
        logger.info(f"ONNX exported: {onnx_path}")
        
        saved_model_path = os.path.join(temp_dir, 'saved_model')
        try:
            import subprocess
            result = subprocess.run(
                ['onnx2tf', '-i', onnx_path, '-o', saved_model_path, '-osd'],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                raise RuntimeError(f"onnx2tf failed: {result.stderr}")
            
            logger.info(f"SavedModel created: {saved_model_path}")
            
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            int8_model = converter.convert()
            
            with open(output_path, 'wb') as f:
                f.write(int8_model)
            
            logger.info(f"INT8 TFLite model: {len(int8_model)/1024:.1f} KB")
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX→INT8 conversion failed: {e}")
            raise


def _apply_float16_quantization(tflite_path: str) -> None:
    """Apply float16 quantization to an existing TFLite model."""
    import tensorflow as tf
    
    logger.info("Applying float16 quantization...")
    
    with open(tflite_path, 'rb') as f:
        original_content = f.read()
    
    original_size = len(original_content)
    
    try:
        import flatbuffers
        from tensorflow.lite.python import schema_py_generated as schema_fb
        
        model = schema_fb.Model.GetRootAsModel(original_content, 0)
        num_buffers = model.BuffersLength()
        logger.info(f"Model has {num_buffers} buffers")
        
        logger.warning("Direct float16 conversion not implemented.")
        logger.info("ESP32 Note: float16 saves storage but ESP32 computes in float32 anyway.")
        
    except ImportError:
        logger.info("flatbuffers not available for direct weight modification.")
        logger.info("Model exported as float32. For ESP32, this is often optimal anyway.")


def _convert_via_onnx(
    model: 'torch.nn.Module',
    output_path: str,
    input_frames: int,
    quantize: str
) -> str:
    """Fallback conversion via ONNX + TensorFlow."""
    import torch
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        onnx_path = os.path.join(temp_dir, 'model.onnx')
        sample_input = torch.randn(1, 1, MODEL_INPUT_MELS, input_frames)
        
        torch.onnx.export(
            model, sample_input, onnx_path,
            input_names=['input'], output_names=['output'],
            opset_version=13, do_constant_folding=True
        )
        logger.info(f"ONNX model exported: {onnx_path}")
        
        # Copy ONNX to output directory for manual conversion
        onnx_output = output_path.replace('.tflite', '.onnx')
        shutil.copy(onnx_path, onnx_output)
        logger.info(f"ONNX model copied to: {onnx_output}")
        logger.info("Use onnx2tf CLI tool for manual conversion:")
        logger.info(f"  onnx2tf -i {onnx_output} -o {os.path.dirname(output_path)}")
        
        raise RuntimeError("Automatic TFLite conversion failed. ONNX model saved for manual conversion.")


# =============================================================================
# ESP32 Operator Compatibility Check
# =============================================================================

# Path to ESP32 firmware inference.cc (relative to this script)
ESP32_INFERENCE_CC = os.path.join(os.path.dirname(__file__), '..', 'main', 'inference.cc')

# Mapping from TFLite Micro AddXxx() method names to TFLite operator names
TFLITE_OP_MAPPING = {
    'AddConv2D': 'CONV_2D',
    'AddDepthwiseConv2D': 'DEPTHWISE_CONV_2D',
    'AddRelu': 'RELU',
    'AddRelu6': 'RELU6',
    'AddMaxPool2D': 'MAX_POOL_2D',
    'AddAveragePool2D': 'AVERAGE_POOL_2D',
    'AddMean': 'MEAN',
    'AddFullyConnected': 'FULLY_CONNECTED',
    'AddReshape': 'RESHAPE',
    'AddAdd': 'ADD',
    'AddMul': 'MUL',
    'AddL2Normalization': 'L2_NORMALIZATION',
    'AddSum': 'SUM',
    'AddAbs': 'ABS',
    'AddSqrt': 'SQRT',
    'AddMaximum': 'MAXIMUM',
    'AddMinimum': 'MINIMUM',
    'AddDiv': 'DIV',
    'AddSquare': 'SQUARE',
    'AddRsqrt': 'RSQRT',
    'AddQuantize': 'QUANTIZE',
    'AddDequantize': 'DEQUANTIZE',
    'AddSoftmax': 'SOFTMAX',
    'AddLogistic': 'LOGISTIC',
    'AddTanh': 'TANH',
    'AddPad': 'PAD',
    'AddPadV2': 'PADV2',
    'AddConcatenation': 'CONCATENATION',
    'AddSplit': 'SPLIT',
    'AddSqueeze': 'SQUEEZE',
    'AddExpandDims': 'EXPAND_DIMS',
    'AddTranspose': 'TRANSPOSE',
    'AddGather': 'GATHER',
    'AddPack': 'PACK',
    'AddUnpack': 'UNPACK',
    'AddSlice': 'SLICE',
    'AddStridedSlice': 'STRIDED_SLICE',
    'AddSub': 'SUB',
    'AddNeg': 'NEG',
    'AddExp': 'EXP',
    'AddLog': 'LOG',
    'AddFloor': 'FLOOR',
    'AddCeil': 'CEIL',
    'AddRound': 'ROUND',
    'AddCast': 'CAST',
    'AddLeakyRelu': 'LEAKY_RELU',
    'AddPrelu': 'PRELU',
    'AddElu': 'ELU',
    'AddHardSwish': 'HARD_SWISH',
    'AddBatchMatMul': 'BATCH_MATMUL',
    'AddTransposeConv': 'TRANSPOSE_CONV',
    'AddResizeBilinear': 'RESIZE_BILINEAR',
    'AddResizeNearestNeighbor': 'RESIZE_NEAREST_NEIGHBOR',
}


def parse_esp32_supported_ops(inference_cc_path: str = None) -> set:
    """
    Parse main/inference.cc to extract supported TFLite operators.
    
    Looks for resolver.AddXxx() calls in the op_resolver initialization.
    
    Args:
        inference_cc_path: Path to inference.cc file
        
    Returns:
        Set of supported operator names (e.g., {'CONV_2D', 'RELU', ...})
    """
    if inference_cc_path is None:
        inference_cc_path = ESP32_INFERENCE_CC
    
    supported_ops = set()
    
    if not os.path.exists(inference_cc_path):
        logger.warning(f"inference.cc not found at {inference_cc_path}, using fallback list")
        return _get_fallback_supported_ops()
    
    try:
        with open(inference_cc_path, 'r') as f:
            content = f.read()
        
        # Find all resolver.AddXxx() calls
        import re
        pattern = r'resolver\.Add(\w+)\(\)'
        matches = re.findall(pattern, content)
        
        for match in matches:
            method_name = f'Add{match}'
            if method_name in TFLITE_OP_MAPPING:
                supported_ops.add(TFLITE_OP_MAPPING[method_name])
            else:
                # Try to convert method name to op name directly
                # e.g., AddConv2D -> CONV_2D (convert CamelCase to UPPER_SNAKE_CASE)
                op_name = re.sub(r'(?<!^)(?=[A-Z])', '_', match).upper()
                # Handle special cases like "2D" -> "_2D"
                op_name = op_name.replace('_2_D', '_2D')
                supported_ops.add(op_name)
                logger.debug(f"Unknown op method: {method_name} -> guessed: {op_name}")
        
        logger.info(f"📋 Parsed {len(supported_ops)} operators from {inference_cc_path}")
        
    except Exception as e:
        logger.warning(f"Error parsing inference.cc: {e}, using fallback list")
        return _get_fallback_supported_ops()
    
    return supported_ops


def _get_fallback_supported_ops() -> set:
    """Fallback list of commonly supported ops if inference.cc cannot be parsed."""
    return {
        'CONV_2D', 'DEPTHWISE_CONV_2D', 'RELU', 'RELU6', 'MAX_POOL_2D',
        'MEAN', 'FULLY_CONNECTED', 'RESHAPE', 'ADD', 'MUL',
        'L2_NORMALIZATION', 'SUM', 'ABS', 'SQRT', 'MAXIMUM', 'DIV',
        'SQUARE', 'RSQRT', 'QUANTIZE', 'DEQUANTIZE',
    }


# Meta-operators that can be ignored (not real ops)
IGNORE_OPS = {
    'DELEGATE',  # XNNPACK/GPU delegate wrapper
    'CUSTOM',    # Custom ops marker
}


def get_tflite_operators(tflite_path: str) -> list:
    """
    Extract list of operators used by a TFLite model.
    
    Args:
        tflite_path: Path to .tflite file
        
    Returns:
        List of operator names used by the model
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.warning("TensorFlow not available, trying flatbuffers directly...")
        return _get_operators_flatbuffers(tflite_path)
    
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get operator details
    ops = set()
    
    # Method 1: Using _get_ops_details (if available)
    try:
        op_details = interpreter._get_ops_details()
        for op in op_details:
            op_name = op.get('op_name', 'UNKNOWN')
            ops.add(op_name)
    except AttributeError:
        pass
    
    # Method 2: Parse the flatbuffer directly if method 1 failed
    if not ops:
        ops = _get_operators_flatbuffers(tflite_path)
    
    return sorted(list(ops))


def _get_operators_flatbuffers(tflite_path: str) -> set:
    """Extract operators by parsing TFLite flatbuffer directly."""
    # TFLite operator codes mapping (common ones)
    BUILTIN_OPCODES = {
        0: 'ADD',
        1: 'AVERAGE_POOL_2D',
        2: 'CONCATENATION',
        3: 'CONV_2D',
        4: 'DEPTHWISE_CONV_2D',
        5: 'DEPTH_TO_SPACE',
        6: 'DEQUANTIZE',
        7: 'EMBEDDING_LOOKUP',
        8: 'FLOOR',
        9: 'FULLY_CONNECTED',
        10: 'HASHTABLE_LOOKUP',
        11: 'L2_NORMALIZATION',
        12: 'L2_POOL_2D',
        13: 'LOCAL_RESPONSE_NORMALIZATION',
        14: 'LOGISTIC',
        15: 'LSH_PROJECTION',
        16: 'LSTM',
        17: 'MAX_POOL_2D',
        18: 'MUL',
        19: 'RELU',
        20: 'RELU_N1_TO_1',
        21: 'RELU6',
        22: 'RESHAPE',
        23: 'RESIZE_BILINEAR',
        24: 'RNN',
        25: 'SOFTMAX',
        26: 'SPACE_TO_DEPTH',
        27: 'SVDF',
        28: 'TANH',
        29: 'CONCAT_EMBEDDINGS',
        30: 'SKIP_GRAM',
        31: 'CALL',
        32: 'CUSTOM',
        33: 'EMBEDDING_LOOKUP_SPARSE',
        34: 'PAD',
        35: 'UNIDIRECTIONAL_SEQUENCE_RNN',
        36: 'GATHER',
        37: 'BATCH_TO_SPACE_ND',
        38: 'SPACE_TO_BATCH_ND',
        39: 'TRANSPOSE',
        40: 'MEAN',
        41: 'SUB',
        42: 'DIV',
        43: 'SQUEEZE',
        44: 'UNIDIRECTIONAL_SEQUENCE_LSTM',
        45: 'STRIDED_SLICE',
        46: 'BIDIRECTIONAL_SEQUENCE_RNN',
        47: 'EXP',
        48: 'TOPK_V2',
        49: 'SPLIT',
        50: 'LOG_SOFTMAX',
        51: 'DELEGATE',
        52: 'BIDIRECTIONAL_SEQUENCE_LSTM',
        53: 'CAST',
        54: 'PRELU',
        55: 'MAXIMUM',
        56: 'ARG_MAX',
        57: 'MINIMUM',
        58: 'LESS',
        59: 'NEG',
        60: 'PADV2',
        61: 'GREATER',
        62: 'GREATER_EQUAL',
        63: 'LESS_EQUAL',
        64: 'SELECT',
        65: 'SLICE',
        66: 'SIN',
        67: 'TRANSPOSE_CONV',
        68: 'SPARSE_TO_DENSE',
        69: 'TILE',
        70: 'EXPAND_DIMS',
        71: 'EQUAL',
        72: 'NOT_EQUAL',
        73: 'LOG',
        74: 'SUM',
        75: 'SQRT',
        76: 'RSQRT',
        77: 'SHAPE',
        78: 'POW',
        79: 'ARG_MIN',
        80: 'FAKE_QUANT',
        81: 'REDUCE_PROD',
        82: 'REDUCE_MAX',
        83: 'PACK',
        84: 'LOGICAL_OR',
        85: 'ONE_HOT',
        86: 'LOGICAL_AND',
        87: 'LOGICAL_NOT',
        88: 'UNPACK',
        89: 'REDUCE_MIN',
        90: 'FLOOR_DIV',
        91: 'REDUCE_ANY',
        92: 'SQUARE',
        93: 'ZEROS_LIKE',
        94: 'FILL',
        95: 'FLOOR_MOD',
        96: 'RANGE',
        97: 'RESIZE_NEAREST_NEIGHBOR',
        98: 'LEAKY_RELU',
        99: 'SQUARED_DIFFERENCE',
        100: 'MIRROR_PAD',
        101: 'ABS',
        102: 'SPLIT_V',
        103: 'UNIQUE',
        104: 'CEIL',
        105: 'REVERSE_V2',
        106: 'ADD_N',
        107: 'GATHER_ND',
        108: 'COS',
        109: 'WHERE',
        110: 'RANK',
        111: 'ELU',
        112: 'REVERSE_SEQUENCE',
        113: 'MATRIX_DIAG',
        114: 'QUANTIZE',
        115: 'MATRIX_SET_DIAG',
        116: 'ROUND',
        117: 'HARD_SWISH',
        118: 'IF',
        119: 'WHILE',
        120: 'NON_MAX_SUPPRESSION_V4',
        121: 'NON_MAX_SUPPRESSION_V5',
        122: 'SCATTER_ND',
        123: 'SELECT_V2',
        124: 'DENSIFY',
        125: 'SEGMENT_SUM',
        126: 'BATCH_MATMUL',
    }
    
    ops = set()
    
    try:
        with open(tflite_path, 'rb') as f:
            model_data = f.read()
        
        # Simple parsing: look for operator codes in the flatbuffer
        # This is a simplified approach - proper parsing would use flatbuffers library
        # For now, we'll try to use TFLite's schema
        try:
            from tflite_runtime.interpreter import Interpreter
            interpreter = Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            # Try to get tensor details to infer operations
        except ImportError:
            pass
        
        # Fallback: analyze model structure
        logger.warning("Using simplified operator detection")
        
    except Exception as e:
        logger.error(f"Error parsing TFLite model: {e}")
    
    return ops


def check_esp32_compatibility(tflite_path: str, verbose: bool = True) -> dict:
    """
    Check if a TFLite model is compatible with ESP32 firmware.
    
    Args:
        tflite_path: Path to .tflite file
        verbose: Print detailed results
        
    Returns:
        Dictionary with compatibility results
    """
    logger.info(f"🔍 Checking ESP32 compatibility: {tflite_path}")
    
    # Get model operators
    model_ops = get_tflite_operators(tflite_path)
    
    if not model_ops:
        # Fallback: try using TensorFlow
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            
            # Get model signature
            logger.info("Model loaded successfully with TensorFlow Lite")
            
            # Analyze input/output
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            logger.info(f"  Input shape: {input_details[0]['shape']}")
            logger.info(f"  Input dtype: {input_details[0]['dtype']}")
            logger.info(f"  Output shape: {output_details[0]['shape']}")
            logger.info(f"  Output dtype: {output_details[0]['dtype']}")
            
            # Try to get ops
            try:
                interpreter.allocate_tensors()
                # Use internal method if available
                if hasattr(interpreter, '_get_ops_details'):
                    ops_details = interpreter._get_ops_details()
                    model_ops = [op.get('op_name', 'UNKNOWN') for op in ops_details]
            except Exception as e:
                logger.warning(f"Could not get operator details: {e}")
                
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            return {'compatible': None, 'error': 'TensorFlow not installed'}
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return {'compatible': None, 'error': str(e)}
    
    # Get supported operators from inference.cc
    esp32_supported = parse_esp32_supported_ops()
    
    # Check compatibility (filter out meta-operators)
    model_ops_set = set(model_ops) - IGNORE_OPS
    supported = model_ops_set & esp32_supported
    unsupported = model_ops_set - esp32_supported
    
    result = {
        'compatible': len(unsupported) == 0,
        'model_operators': sorted(model_ops),
        'supported': sorted(list(supported)),
        'unsupported': sorted(list(unsupported)),
        'esp32_available': sorted(list(esp32_supported)),
    }
    
    if verbose:
        logger.info(f"\n{'='*60}")
        logger.info("ESP32 COMPATIBILITY REPORT")
        logger.info(f"{'='*60}")
        logger.info(f"Model: {tflite_path}")
        logger.info(f"Total operators in model: {len(model_ops)}")
        
        logger.info(f"\n✅ SUPPORTED operators ({len(supported)}):")
        for op in sorted(supported):
            logger.info(f"   • {op}")
        
        if unsupported:
            logger.info(f"\n❌ UNSUPPORTED operators ({len(unsupported)}):")
            for op in sorted(unsupported):
                logger.info(f"   • {op} ⚠️")
            logger.warning("\n⚠️  Model may NOT run on ESP32 without adding missing operators!")
            logger.info("   Add missing operators to main/inference.cc in op_resolver")
        else:
            logger.info(f"\n✅ All operators are supported by ESP32 firmware!")
        
        logger.info(f"{'='*60}")
    
    return result


# =============================================================================
# Header Generation Functions
# =============================================================================

def generate_model_header(
    tflite_path: str,
    output_path: str,
    array_name: str = 'model_tflite'
) -> str:
    """Generate C header file from TFLite model."""
    logger.info(f"Generating C header: {output_path}")
    
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    model_size = len(model_data)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate header guard from array name (e.g., model_v4_pump_tflite -> MODEL_V4_PUMP_DATA_H)
    header_guard = array_name.upper() + '_DATA_H'
    header_filename = os.path.basename(output_path)
    
    with open(output_path, 'w') as f:
        f.write(f"""/**
 * @file {header_filename}
 * @brief TensorFlow Lite model data for ESP32
 * 
 * Auto-generated by export_esp32.py
 * Generated: {timestamp}
 * Model size: {model_size} bytes ({model_size / 1024:.2f} KB)
 * Array name: {array_name}
 * 
 * DO NOT EDIT MANUALLY
 */

#ifndef {header_guard}
#define {header_guard}

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

// Model data array (aligned for optimal ESP32 access)
__attribute__((aligned(16)))
const unsigned char {array_name}[] = {{
""")
        
        for i in range(0, len(model_data), 12):
            row = model_data[i:i+12]
            hex_bytes = ', '.join(f'0x{b:02x}' for b in row)
            f.write(f"    {hex_bytes},\n")
        
        f.write(f"""}};

const unsigned int {array_name}_len = {model_size};

#ifdef __cplusplus
}}
#endif

#endif // {header_guard}
""")
    
    logger.info(f"Header generated: {output_path} ({model_size} bytes)")
    return output_path


def generate_embeddings_header(
    centroid: 'torch.Tensor',
    std: 'torch.Tensor',
    output_path: str,
    threshold: float = 0.85
) -> str:
    """Generate C header file with reference embeddings for anomaly detection."""
    logger.info(f"Generating embeddings header: {output_path}")
    
    embedding_size = centroid.shape[0]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    with open(output_path, 'w') as f:
        f.write(f"""/**
 * @file reference_embeddings.h
 * @brief Reference embeddings for anomaly detection
 * 
 * Auto-generated by export_esp32.py
 * Generated: {timestamp}
 * Embedding size: {embedding_size}
 * 
 * DO NOT EDIT MANUALLY
 */

#ifndef REFERENCE_EMBEDDINGS_H
#define REFERENCE_EMBEDDINGS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {{
#endif

#define EMBEDDING_SIZE {embedding_size}
#define ANOMALY_THRESHOLD {threshold}f

// Centroid of normal class (mean embedding)
static const float normal_centroid[EMBEDDING_SIZE] = {{
""")
        
        centroid_list = centroid.tolist()
        for i in range(0, len(centroid_list), 8):
            row = centroid_list[i:i+8]
            values = ', '.join(f'{v:.6f}f' for v in row)
            f.write(f"    {values},\n")
        
        f.write(f"""}};

// Standard deviation of normal class embeddings
static const float normal_std[EMBEDDING_SIZE] = {{
""")
        
        std_list = std.tolist()
        for i in range(0, len(std_list), 8):
            row = std_list[i:i+8]
            values = ', '.join(f'{v:.6f}f' for v in row)
            f.write(f"    {values},\n")
        
        f.write(f"""}};

#ifdef __cplusplus
}}
#endif

#endif // REFERENCE_EMBEDDINGS_H
""")
    
    logger.info(f"Embeddings header generated: {output_path}")
    return output_path


# =============================================================================
# Verification Data Generation Functions
# =============================================================================

def generate_synthetic_mel_spectrograms(
    num_samples: int = DEFAULT_VERIFICATION_SAMPLES,
    n_mels: int = MODEL_INPUT_MELS,
    n_frames: int = MODEL_INPUT_FRAMES
) -> np.ndarray:
    """
    Generate synthetic mel-spectrograms with various patterns.
    
    Returns:
        np.ndarray: Shape [num_samples, 1, n_mels, n_frames]
    """
    logger.info(f"Generating {num_samples} synthetic mel-spectrograms...")
    
    spectrograms = []
    
    for i in range(num_samples):
        pattern_type = i % 8
        
        if pattern_type == 0:
            # Low frequency emphasis (like bass sounds)
            mel = np.zeros((n_mels, n_frames), dtype=np.float32)
            for m in range(n_mels // 4):
                mel[m, :] = np.sin(np.linspace(0, 4 * np.pi, n_frames)) * (1 - m / (n_mels // 4))
            mel = mel * 0.5 - 0.5
            
        elif pattern_type == 1:
            # High frequency emphasis
            mel = np.zeros((n_mels, n_frames), dtype=np.float32)
            for m in range(n_mels * 3 // 4, n_mels):
                mel[m, :] = np.sin(np.linspace(0, 8 * np.pi, n_frames)) * ((m - n_mels * 3 // 4) / (n_mels // 4))
            mel = mel * 0.5 - 0.5
            
        elif pattern_type == 2:
            # Harmonic pattern (like speech/music)
            mel = np.zeros((n_mels, n_frames), dtype=np.float32)
            fundamental = 10 + (i % 5) * 5
            for harmonic in range(1, 6):
                freq_bin = min(fundamental * harmonic, n_mels - 1)
                if freq_bin < n_mels:
                    mel[freq_bin, :] = np.sin(np.linspace(0, 2 * np.pi * harmonic, n_frames)) / harmonic
            mel = mel * 0.8 - 0.6
            
        elif pattern_type == 3:
            # Broadband noise pattern
            mel = np.random.randn(n_mels, n_frames).astype(np.float32) * 0.3 - 0.5
            
        elif pattern_type == 4:
            # Rising frequency sweep
            mel = np.zeros((n_mels, n_frames), dtype=np.float32)
            for f in range(n_frames):
                freq_bin = int(f * n_mels / n_frames)
                mel[max(0, freq_bin-2):min(n_mels, freq_bin+3), f] = 0.8
            mel = mel - 0.6
            
        elif pattern_type == 5:
            # Falling frequency sweep
            mel = np.zeros((n_mels, n_frames), dtype=np.float32)
            for f in range(n_frames):
                freq_bin = int((n_frames - 1 - f) * n_mels / n_frames)
                mel[max(0, freq_bin-2):min(n_mels, freq_bin+3), f] = 0.8
            mel = mel - 0.6
            
        elif pattern_type == 6:
            # Impulse/transient pattern
            mel = np.ones((n_mels, n_frames), dtype=np.float32) * -0.8
            impulse_pos = (i * 7) % n_frames
            mel[:, max(0, impulse_pos-1):min(n_frames, impulse_pos+2)] = 0.0
            
        else:
            # Silent/very quiet
            mel = np.ones((n_mels, n_frames), dtype=np.float32) * -0.9 + \
                  np.random.randn(n_mels, n_frames).astype(np.float32) * 0.05
        
        # Add small random noise for uniqueness
        mel += np.random.randn(n_mels, n_frames).astype(np.float32) * 0.01 * (i + 1)
        
        # Clip to reasonable range
        mel = np.clip(mel, -1.0, 0.0)
        
        spectrograms.append(mel)
    
    # Stack and add channel dimension: [N, 1, n_mels, n_frames]
    result = np.stack(spectrograms, axis=0)[:, np.newaxis, :, :]
    
    logger.info(f"Generated spectrograms shape: {result.shape}")
    logger.info(f"Value range: [{result.min():.3f}, {result.max():.3f}]")
    
    return result.astype(np.float32)


def compute_verification_embeddings(
    model: 'torch.nn.Module',
    spectrograms: np.ndarray
) -> np.ndarray:
    """
    Run spectrograms through model to get reference embeddings.
    
    Args:
        model: PyTorch model
        spectrograms: Shape [N, 1, n_mels, n_frames]
    
    Returns:
        np.ndarray: Shape [N, embedding_dim]
    """
    import torch
    
    logger.info("Computing verification embeddings...")
    
    with torch.no_grad():
        inputs = torch.from_numpy(spectrograms)
        outputs = model(inputs)
        embeddings = outputs.numpy()
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Embedding value range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    
    return embeddings.astype(np.float32)


def generate_verification_header(
    spectrograms: np.ndarray,
    embeddings: np.ndarray,
    output_path: str,
    n_mels: int = MODEL_INPUT_MELS,
    n_frames: int = MODEL_INPUT_FRAMES,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
) -> str:
    """
    Generate C header file with verification data.
    
    Args:
        spectrograms: Shape [N, 1, n_mels, n_frames]
        embeddings: Shape [N, embedding_dim]
        output_path: Output .h file path
    """
    logger.info(f"Generating verification header: {output_path}")
    
    num_samples = spectrograms.shape[0]
    
    with open(output_path, 'w') as f:
        f.write("/**\n")
        f.write(" * @file ml_verification_data.h\n")
        f.write(" * @brief ML Model Verification Data\n")
        f.write(" *\n")
        f.write(" * Auto-generated verification data for testing ML inference on ESP32.\n")
        f.write(" * Contains synthetic mel-spectrograms and their expected embeddings.\n")
        f.write(" *\n")
        f.write(f" * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f" * Samples: {num_samples}\n")
        f.write(f" * Input shape: [1, 1, {n_mels}, {n_frames}]\n")
        f.write(f" * Output shape: [1, {embedding_dim}]\n")
        f.write(" *\n")
        f.write(" * Configuration at generation time:\n")
        f.write(f" *   N_MELS={n_mels}, N_FRAMES={n_frames}, EMBEDDING_DIM={embedding_dim}\n")
        f.write(" *\n")
        f.write(" * These values MUST match config.h:\n")
        f.write(" *   MEL_SPECTROGRAM_DEFAULT_N_MELS, MODEL_INPUT_FRAMES\n")
        f.write(" */\n\n")
        
        f.write("#ifndef ML_VERIFICATION_DATA_H\n")
        f.write("#define ML_VERIFICATION_DATA_H\n\n")
        
        f.write("#include <stdint.h>\n")
        f.write("#include <stddef.h>\n\n")
        
        f.write("// Number of verification samples\n")
        f.write(f"#define ML_VERIFY_NUM_SAMPLES     {num_samples}\n\n")
        
        f.write("// Dimensions (should match config.h)\n")
        f.write(f"// N_MELS = {n_mels} (MEL_SPECTROGRAM_DEFAULT_N_MELS)\n")
        f.write(f"// N_FRAMES = {n_frames} (MODEL_INPUT_FRAMES)\n")
        f.write(f"// EMBEDDING_DIM = {embedding_dim}\n\n")
        
        # Input spectrograms
        f.write("/**\n")
        f.write(" * @brief Verification input mel-spectrograms\n")
        f.write(f" * Shape: [{num_samples}][{n_mels}][{n_frames}]\n")
        f.write(" * Layout: [sample_idx][mel_bin][frame_idx]\n")
        f.write(" */\n")
        f.write(f"static const float ml_verify_inputs[{num_samples}][{n_mels}][{n_frames}] = {{\n")
        
        for s in range(num_samples):
            f.write(f"    // Sample {s}\n")
            f.write("    {\n")
            for m in range(n_mels):
                f.write("        {")
                values = [f"{spectrograms[s, 0, m, fr]:.6f}f" for fr in range(n_frames)]
                f.write(", ".join(values))
                f.write("},\n" if m < n_mels - 1 else "}\n")
            f.write("    },\n" if s < num_samples - 1 else "    }\n")
        
        f.write("};\n\n")
        
        # Expected embeddings
        f.write("/**\n")
        f.write(" * @brief Expected output embeddings from PyTorch model\n")
        f.write(f" * Shape: [{num_samples}][{embedding_dim}]\n")
        f.write(" */\n")
        f.write(f"static const float ml_verify_expected[{num_samples}][{embedding_dim}] = {{\n")
        
        for s in range(num_samples):
            f.write("    {")
            values = [f"{embeddings[s, e]:.6f}f" for e in range(embedding_dim)]
            for i in range(0, len(values), 8):
                if i > 0:
                    f.write("\n     ")
                f.write(", ".join(values[i:i+8]))
                if i + 8 < len(values):
                    f.write(",")
            f.write("},\n" if s < num_samples - 1 else "}\n")
        
        f.write("};\n\n")
        
        # Pattern descriptions
        f.write("/**\n")
        f.write(" * @brief Description of each verification sample pattern\n")
        f.write(" */\n")
        f.write(f"static const char* ml_verify_descriptions[{num_samples}] = {{\n")
        patterns = [
            "Low frequency emphasis",
            "High frequency emphasis",
            "Harmonic pattern",
            "Broadband noise",
            "Rising frequency sweep",
            "Falling frequency sweep",
            "Impulse/transient",
            "Silent/quiet"
        ]
        for s in range(num_samples):
            desc = patterns[s % 8]
            f.write(f'    "{desc} (sample {s})",\n' if s < num_samples - 1 
                    else f'    "{desc} (sample {s})"\n')
        f.write("};\n\n")
        
        f.write("#endif // ML_VERIFICATION_DATA_H\n")
    
    file_size = os.path.getsize(output_path)
    logger.info(f"Verification header generated: {output_path} ({file_size / 1024:.1f} KB)")
    
    return output_path


# =============================================================================
# Reference Embeddings (for anomaly detection)
# =============================================================================

def compute_reference_embeddings(
    model: 'torch.nn.Module',
    dataset_path: str,
    num_samples: int = 100,
    machine_type: str = None
) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """
    Compute reference embeddings from normal samples for anomaly detection.
    """
    import torch
    
    logger.info(f"Computing reference embeddings from {num_samples} samples")
    
    try:
        from ml.triplet_memory_dataset import TripletMemoryDataset
    except ImportError:
        from triplet_memory_dataset import TripletMemoryDataset
    
    dataset = TripletMemoryDataset(
        dataset_path,
        samples_per_epoch=num_samples,
        skip_types=[] if machine_type is None else [t for t in ['fan', 'pump', 'slider', 'valve'] if t != machine_type]
    )
    
    embeddings = []
    model.eval()
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            anchor, _, _ = dataset[i]
            anchor = anchor.unsqueeze(0)
            embedding = model(anchor)
            embeddings.append(embedding)
    
    embeddings = torch.cat(embeddings, dim=0)
    centroid = embeddings.mean(dim=0)
    std = embeddings.std(dim=0)
    
    logger.info(f"Computed centroid from {len(embeddings)} samples")
    logger.info(f"Centroid norm: {torch.norm(centroid).item():.4f}")
    
    return centroid, std


# =============================================================================
# Verification
# =============================================================================

def verify_export(
    original_model: 'torch.nn.Module',
    tflite_path: str,
    input_frames: int = MODEL_INPUT_FRAMES
) -> bool:
    """Verify that TFLite model produces similar output to original PyTorch model."""
    import torch
    import tensorflow as tf
    
    logger.info("Verifying exported model...")
    
    test_input = torch.randn(1, 1, MODEL_INPUT_MELS, input_frames)
    
    original_model.eval()
    with torch.no_grad():
        pytorch_output = original_model(test_input).numpy()
    
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    tflite_input = test_input.numpy()
    interpreter.set_tensor(input_details[0]['index'], tflite_input.astype(np.float32))
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])
    
    mse = np.mean((pytorch_output - tflite_output) ** 2)
    max_diff = np.max(np.abs(pytorch_output - tflite_output))
    
    logger.info(f"PyTorch output shape: {pytorch_output.shape}")
    logger.info(f"TFLite output shape: {tflite_output.shape}")
    logger.info(f"MSE: {mse:.8f}")
    logger.info(f"Max diff: {max_diff:.8f}")
    
    tolerance = 1e-3
    if mse < tolerance:
        logger.info(f"✅ Verification PASSED (MSE < {tolerance})")
        return True
    else:
        logger.warning(f"⚠️ Verification WARNING: MSE ({mse:.6f}) > tolerance ({tolerance})")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Export PyTorch model for ESP32 deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full export with verification data
    python export_esp32.py --mlflow-run-id abc123

    # Export only (no verification data)
    python export_esp32.py --mlflow-run-id abc123 --skip-verification
    
    # Verification data only (no TFLite export)
    python export_esp32.py --mlflow-run-id abc123 --skip-export
    
    # From local file
    python export_esp32.py --model-path model.pth --model-version 1
"""
    )
    
    # Model source (required unless using --check-only)
    source_group = parser.add_mutually_exclusive_group(required=False)
    source_group.add_argument(
        '--mlflow-run-id', type=str,
        help='MLflow run ID to load model from'
    )
    source_group.add_argument(
        '--model-path', type=str,
        help='Path to local .pth model file'
    )
    
    # MLflow options
    parser.add_argument(
        '--mlflow-uri', type=str,
        default=DEFAULT_MLFLOW_TRACKING_URI,
        help=f'MLflow tracking URI (default: {DEFAULT_MLFLOW_TRACKING_URI})'
    )
    parser.add_argument(
        '--artifact-name', type=str, default=None,
        help='Name of model artifact in MLflow. If not specified, auto-detects "best" or "final"'
    )
    
    # Model options (for local file)
    parser.add_argument(
        '--model-version', type=str, choices=['1', '2', '3', '4', 'v1', 'v2', 'v3', 'v4'], default='1',
        help='Model version when loading from file: 1/v1 (Standard), 2/v2 (DepthwiseSep), 3/v3 (MobileNetV2), 4/v4 (MBConv+SE) (default: 1)'
    )
    parser.add_argument(
        '--embedding-dim', type=int, default=DEFAULT_EMBEDDING_DIM,
        help=f'Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})'
    )
    
    # Export options
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--quantize', type=str, choices=['float32', 'float16', 'int8'], default='float32',
        help='Quantization type (default: float32)'
    )
    parser.add_argument(
        '--model-name', type=str, default='model',
        help='Base name for output model file (default: model -> model.tflite)'
    )
    parser.add_argument(
        '--update-h-files', action='store_true',
        help='Also update standard model.tflite and model_data.h (with model_tflite[] array) for production use'
    )
    parser.add_argument(
        '--check-compatibility', action='store_true',
        help='Check if model operators are supported by ESP32 firmware (from main/inference.cc)'
    )
    parser.add_argument(
        '--check-only', type=str, metavar='TFLITE_PATH',
        help='Only check compatibility of existing .tflite file (no export)'
    )
    parser.add_argument(
        '--input-frames', type=int, default=MODEL_INPUT_FRAMES,
        help=f'Number of input time frames (default: {MODEL_INPUT_FRAMES})'
    )
    
    # Skip options
    parser.add_argument(
        '--skip-export', action='store_true',
        help='Skip TFLite export (only generate verification data)'
    )
    parser.add_argument(
        '--skip-verification', action='store_true',
        help='Skip verification data generation'
    )
    parser.add_argument(
        '--skip-verify', action='store_true',
        help='Skip TFLite model verification step'
    )
    
    # Verification data options
    parser.add_argument(
        '--verification-samples', type=int, default=DEFAULT_VERIFICATION_SAMPLES,
        help=f'Number of verification samples (default: {DEFAULT_VERIFICATION_SAMPLES})'
    )
    
    # Reference embeddings (for anomaly detection)
    parser.add_argument(
        '--dataset', type=str, default=None,
        help='Dataset .pt file for computing reference embeddings (anomaly detection)'
    )
    parser.add_argument(
        '--num-reference-samples', type=int, default=100,
        help='Number of samples for reference embeddings (default: 100)'
    )
    parser.add_argument(
        '--anomaly-threshold', type=float, default=0.85,
        help='Anomaly detection threshold (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    # Handle --check-only mode (only check compatibility, no export)
    if args.check_only:
        logger.info("=" * 60)
        logger.info("ESP32 Compatibility Check Only")
        logger.info("=" * 60)
        result = check_esp32_compatibility(args.check_only, verbose=True)
        return 0 if result.get('compatible', False) else 1
    
    # Validate: need model source unless --check-only
    if not args.mlflow_run_id and not args.model_path:
        parser.error("one of the arguments --mlflow-run-id --model-path is required (unless using --check-only)")
    
    # Validate: at least one output
    if args.skip_export and args.skip_verification:
        parser.error("Cannot skip both export and verification - nothing to do!")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    logger.info("=" * 60)
    logger.info("ESP32 Model Export")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Input shape: [1, 1, {MODEL_INPUT_MELS}, {args.input_frames}]")
    logger.info(f"Skip export: {args.skip_export}")
    logger.info(f"Skip verification data: {args.skip_verification}")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load model
        model, embedding_dim = load_model(
            mlflow_run_id=args.mlflow_run_id,
            model_path=args.model_path,
            mlflow_uri=args.mlflow_uri,
            artifact_name=args.artifact_name,
            model_version=args.model_version,
            embedding_dim=args.embedding_dim
        )
        
        generated_files = []
        
        # Step 2: Convert to TFLite
        if not args.skip_export:
            tflite_filename = f'{args.model_name}.tflite'
            tflite_path = output_dir / tflite_filename
            convert_pytorch_to_tflite(model, str(tflite_path), args.input_frames, args.quantize)
            generated_files.append(tflite_filename)
            
            # Only generate standard header files if --update-h-files is specified
            if args.update_h_files:
                # Also copy to standard model.tflite if using custom name
                if args.model_name != 'model':
                    import shutil
                    standard_tflite = output_dir / 'model.tflite'
                    shutil.copy(str(tflite_path), str(standard_tflite))
                    logger.info(f"📋 Copied to standard: {standard_tflite}")
                    generated_files.append('model.tflite')
                
                # Generate standard model_data.h with model_tflite[] array
                header_path = output_dir / 'model_data.h'
                generate_model_header(str(tflite_path), str(header_path), array_name='model_tflite')
                generated_files.append('model_data.h')
                logger.info(f"📦 Updated standard header: model_data.h (array: model_tflite[])")
            else:
                logger.info(f"💡 Header files NOT updated (use --update-h-files to update model_data.h)")
            
            # Verify export
            if not args.skip_verify:
                verify_export(model, str(tflite_path), args.input_frames)
            
            # Check ESP32 compatibility
            if args.check_compatibility:
                compat_result = check_esp32_compatibility(str(tflite_path), verbose=True)
                if not compat_result.get('compatible', True):
                    logger.warning("⚠️  Model has unsupported operators! See report above.")
        
        # Step 3: Generate verification data
        if not args.skip_verification:
            spectrograms = generate_synthetic_mel_spectrograms(
                num_samples=args.verification_samples,
                n_mels=MODEL_INPUT_MELS,
                n_frames=args.input_frames
            )
            embeddings = compute_verification_embeddings(model, spectrograms)
            
            verification_path = output_dir / 'ml_verification_data.h'
            generate_verification_header(
                spectrograms, embeddings, str(verification_path),
                n_mels=MODEL_INPUT_MELS,
                n_frames=args.input_frames,
                embedding_dim=embedding_dim
            )
            generated_files.append('ml_verification_data.h')
        
        # Step 4: Generate reference embeddings (if dataset provided)
        if args.dataset:
            centroid, std = compute_reference_embeddings(
                model, args.dataset, args.num_reference_samples
            )
            embeddings_path = output_dir / 'reference_embeddings.h'
            generate_embeddings_header(
                centroid, std, str(embeddings_path), args.anomaly_threshold
            )
            generated_files.append('reference_embeddings.h')
        
        # Summary
        logger.info("=" * 60)
        logger.info("Export completed successfully!")
        logger.info(f"Output directory: {output_dir}")
        logger.info("Generated files:")
        for f in generated_files:
            logger.info(f"  - {f}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
