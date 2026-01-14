#!/usr/bin/env python3
"""
Embedding Visualization Module for Audio Anomaly Detection

Simple module for visualizing batches of embedding vectors in 3D space.
Each batch is displayed in a different color, allowing comparison of multiple samples.

Usage:
    from visualize_embeddings import visualize_batches
    
    # List of batches (each batch is a numpy array or torch tensor)
    batches = [
        embeddings_sample1,  # shape: [N1, embedding_dim]
        embeddings_sample2,  # shape: [N2, embedding_dim]
        embeddings_sample3,  # shape: [N3, embedding_dim]
    ]
    
    visualize_batches(
        batches=batches,
        labels=['Sample 1', 'Sample 2', 'Sample 3'],
        output_file='embeddings.png'
    )

Jupyter Notebook Usage:
    For displaying plots in Jupyter notebooks, use one of these magic commands:
    
    # Static plots (embedded in notebook):
    %matplotlib inline
    
    # Interactive 3D plots (can rotate, zoom, pan):
    # First install: pip install ipympl
    # Then use:
    %matplotlib widget
    
    # Then call visualize_batches() normally - the plot will appear interactive:
    fig = visualize_batches(batches, labels)
    
    # With widget backend, you can:
    # - Rotate: Click and drag with mouse
    # - Zoom: Scroll wheel
    # - Pan: Right-click and drag
"""

import numpy as np
import os
import argparse
import logging
import tempfile
from typing import List, Optional, Union, Tuple
import torch

# Matplotlib imports (optional, checked at runtime)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    Axes3D = None

# Try to import sklearn for dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import mlflow for artifact logging
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

# Constants
DEFAULT_REDUCTION_METHOD = os.getenv('VISUALIZATION_REDUCTION_METHOD', 'pca')


def _to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert torch tensor to numpy array if needed."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def reduce_dimensions(
    embeddings: np.ndarray,
    method: str = 'pca',
    n_components: int = 3,
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embedding dimensions to 3D using PCA or t-SNE.
    
    Args:
        embeddings: Embedding array of shape [N, embedding_dim]
        method: Reduction method ('pca' or 'tsne')
        n_components: Number of components (should be 3 for 3D visualization)
        random_state: Random state for reproducibility
    
    Returns:
        Reduced embeddings of shape [N, 3]
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for dimensionality reduction. Install it with: pip install scikit-learn")
    
    if embeddings.shape[1] <= 3:
        logger.info(f"Embeddings already have {embeddings.shape[1]} dimensions, padding to 3D if needed")
        # Pad to 3D if needed
        if embeddings.shape[1] < 3:
            padding = np.zeros((embeddings.shape[0], 3 - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        return embeddings
    
    logger.info(f"Reducing dimensions from {embeddings.shape[1]} to {n_components} using {method.upper()}...")
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        explained_variance = reducer.explained_variance_ratio_
        logger.info(f"PCA explained variance: {explained_variance}")
        logger.info(f"Total explained variance: {sum(explained_variance):.2%}")
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
        reduced = reducer.fit_transform(embeddings)
        logger.info("t-SNE reduction completed")
    else:
        raise ValueError(f"Unknown reduction method: {method}. Use 'pca' or 'tsne'")
    
    return reduced


def visualize_batches(
    batches: List[Union[np.ndarray, torch.Tensor]],
    labels: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    reduction_method: str = DEFAULT_REDUCTION_METHOD,
    n_components: int = 3,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    alpha: float = 0.6,
    s: int = 20,
    random_state: int = 42,
    save_to_mlflow: bool = False,
    mlflow_run_id: Optional[str] = None,
    mlflow_artifact_path: str = "visualizations"
) -> object:
    """
    Visualize multiple batches of embedding vectors in 2D or 3D space.
    
    Each batch is displayed in a different color, allowing comparison of multiple samples.
    
    Args:
        batches: List of batches, where each batch is a numpy array or torch tensor
                 of shape [N, embedding_dim]. Each batch represents vectors from one sample.
        labels: Optional list of labels for each batch (default: 'Batch 0', 'Batch 1', ...)
        output_file: Path to save visualization (None = don't save)
        reduction_method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of dimensions for visualization (2 or 3, default: 3)
        show_plot: Whether to display the plot (in Jupyter, set to False and return fig for widget mode)
        figsize: Figure size (width, height)
        alpha: Transparency of points (0.0 to 1.0)
        s: Size of points
        random_state: Random state for reproducibility
    
    Example:
        # Compare embeddings from 3 different samples
        batches = [
            embeddings_sample1,  # [50, 64] - 50 vectors from sample 1
            embeddings_sample2,  # [30, 64] - 30 vectors from sample 2
            embeddings_sample3,  # [40, 64] - 40 vectors from sample 3
        ]
        visualize_batches(
            batches=batches,
            labels=['Normal Sample', 'Anomaly Sample 1', 'Anomaly Sample 2'],
            output_file='comparison.png'
        )
    
    Jupyter Notebook:
        # For static plots (embedded in notebook):
        %matplotlib inline
        
        # For interactive 3D plots (can rotate, zoom, pan):
        # First install: pip install ipympl
        # Then use:
        %matplotlib widget
        
        # Then call the function normally - plot will be interactive:
        fig = visualize_batches(batches=batches, labels=labels)
        
        # With widget backend, you can:
        # - Rotate: Click and drag with mouse
        # - Zoom: Scroll wheel
        # - Pan: Right-click and drag
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for dimensionality reduction. Install it with: pip install scikit-learn")
    
    if len(batches) == 0:
        raise ValueError("batches list cannot be empty")
    
    # Convert all batches to numpy and validate
    numpy_batches = []
    for i, batch in enumerate(batches):
        batch_np = _to_numpy(batch)
        if len(batch_np.shape) != 2:
            raise ValueError(f"Batch {i} must be 2D array [N, embedding_dim], got shape {batch_np.shape}")
        if batch_np.shape[0] == 0:
            logger.warning(f"Batch {i} is empty, skipping")
            continue
        numpy_batches.append(batch_np)
    
    if len(numpy_batches) == 0:
        raise ValueError("All batches are empty")
    
    # Check that all batches have the same embedding dimension
    embedding_dim = numpy_batches[0].shape[1]
    for i, batch in enumerate(numpy_batches):
        if batch.shape[1] != embedding_dim:
            raise ValueError(f"Batch {i} has embedding_dim={batch.shape[1]}, expected {embedding_dim}")
    
    # Generate labels if not provided
    if labels is None:
        labels = [f'Batch {i}' for i in range(len(numpy_batches))]
    elif len(labels) != len(numpy_batches):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of batches ({len(numpy_batches)})")
    
    if n_components not in [2, 3]:
        raise ValueError(f"n_components must be 2 or 3, got {n_components}")
    
    dim_str = f"{n_components}D"
    logger.info(f"Visualizing {len(numpy_batches)} batches in {dim_str} space")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Reduction method: {reduction_method}")
    
    # Combine all batches for dimensionality reduction
    # This ensures consistent scaling across batches
    all_embeddings = np.vstack(numpy_batches)
    logger.info(f"Total vectors: {all_embeddings.shape[0]}")
    
    # Reduce dimensions
    all_embeddings_reduced = reduce_dimensions(
        all_embeddings,
        method=reduction_method,
        n_components=n_components,
        random_state=random_state
    )
    
    # Split back into batches
    batch_reduced_list = []
    start_idx = 0
    for batch in numpy_batches:
        end_idx = start_idx + batch.shape[0]
        batch_reduced_list.append(all_embeddings_reduced[start_idx:end_idx])
        start_idx = end_idx
    
    # Create plot (2D or 3D)
    fig = plt.figure(figsize=figsize)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
    
    # Generate high-contrast colors for each batch
    # Use a custom color palette with high contrast colors
    high_contrast_colors = [
        '#FF0000',  # Red
        '#0000FF',  # Blue
        '#00FF00',  # Green
        '#FF00FF',  # Magenta
        '#FFFF00',  # Yellow
        '#00FFFF',  # Cyan
        '#FF8000',  # Orange
        '#8000FF',  # Purple
        '#FF0080',  # Pink
        '#0080FF',  # Light Blue
        '#80FF00',  # Lime
        '#FF4000',  # Red-Orange
        '#4000FF',  # Dark Blue
        '#00FF80',  # Spring Green
        '#FFA500',  # Orange (different shade)
    ]
    
    # Use high-contrast colors, cycling if needed
    if len(batch_reduced_list) <= len(high_contrast_colors):
        colors = high_contrast_colors[:len(batch_reduced_list)]
    else:
        # If more batches than colors, use tab20 colormap as fallback
        colors = plt.cm.tab20(np.linspace(0, 1, len(batch_reduced_list)))
    
    # Plot each batch
    for i, (batch_reduced, label, color) in enumerate(zip(batch_reduced_list, labels, colors)):
        if n_components == 3:
            ax.scatter(
                batch_reduced[:, 0],
                batch_reduced[:, 1],
                batch_reduced[:, 2],
                c=[color],
                label=label,
                alpha=alpha,
                s=s
            )
        else:
            ax.scatter(
                batch_reduced[:, 0],
                batch_reduced[:, 1],
                c=[color],
                label=label,
                alpha=alpha,
                s=s
            )
        logger.info(f"  Batch {i} ({label}): {batch_reduced.shape[0]} vectors")
    
    ax.set_xlabel(f'{reduction_method.upper()} Component 1')
    ax.set_ylabel(f'{reduction_method.upper()} Component 2')
    if n_components == 3:
        ax.set_zlabel(f'{reduction_method.upper()} Component 3')
    ax.set_title(f'{dim_str} Embedding Visualization ({len(batch_reduced_list)} batches)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.info(f"✅ Visualization saved to: {output_file}")
    
    # Save to MLflow if requested
    mlflow_artifact_file = None
    if save_to_mlflow and MLFLOW_AVAILABLE:
        try:
            if mlflow_run_id is None:
                logger.warning("save_to_mlflow=True but mlflow_run_id not provided. Cannot save to MLflow.")
            else:
                # Create temporary file for the plot
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                    mlflow_artifact_file = f.name
                
                # Save figure to temporary file
                fig.savefig(mlflow_artifact_file, dpi=150, bbox_inches='tight')
                
                # Start MLflow run if needed
                active_run = mlflow.active_run()
                started_run = False
                
                if not active_run:
                    mlflow.start_run(run_id=mlflow_run_id)
                    started_run = True
                elif active_run.info.run_id != mlflow_run_id:
                    mlflow.end_run()
                    mlflow.start_run(run_id=mlflow_run_id)
                    started_run = True
                
                # Log artifact
                mlflow.log_artifact(mlflow_artifact_file, mlflow_artifact_path)
                logger.info(f"✅ Visualization saved to MLflow: {mlflow_artifact_path}/{os.path.basename(mlflow_artifact_file)}")
                
                # Close run if we started it
                if started_run:
                    mlflow.end_run()
        except Exception as e:
            logger.warning(f"Failed to save visualization to MLflow: {e}")
        finally:
            # Clean up temporary file
            if mlflow_artifact_file and os.path.exists(mlflow_artifact_file):
                try:
                    os.unlink(mlflow_artifact_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")
    
    # For Jupyter notebook widget mode, don't call plt.show() - just return the figure
    # For inline mode or script mode, show the plot
    if show_plot:
        # Check if we're in Jupyter notebook with widget backend
        try:
            import matplotlib
            backend = matplotlib.get_backend()
            if 'widget' in backend.lower() or 'ipympl' in backend.lower():
                # Widget mode: don't call plt.show(), just return fig
                pass
            else:
                # Inline or other mode: show the plot
                plt.show()
        except:
            # Fallback: show the plot
            plt.show()
    else:
        plt.close()
    
    logger.info("✅ Visualization complete!")
    
    # Handle display based on backend
    # For interactive backends (widget, notebook), don't call plt.show()
    # For non-interactive backends, call plt.show() if requested
    backend = plt.get_backend().lower()
    is_interactive_backend = 'widget' in backend or 'notebook' in backend or 'qt' in backend or 'tk' in backend
    
    if show_plot:
        if is_interactive_backend:
            # For interactive backends, just return the figure
            # The figure will be displayed automatically in Jupyter
            logger.debug(f"Using interactive backend: {backend}, figure will be displayed automatically")
        else:
            # For non-interactive backends, call plt.show()
            plt.show()
    
    # Return figure for Jupyter notebook display (especially widget mode)
    return fig


def visualize_wav_files(
    model_path: str,
    wav_files: List[str],
    labels: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    sample_rate: int = 16000,
    duration_sec: float = 1.0,
    chunks_per_file: int = 50,
    rms_threshold: float = 0.005,  # Default RMS threshold for silence filtering
    embedding_dim: int = 64,
    output_file: Optional[str] = None,
    reduction_method: str = DEFAULT_REDUCTION_METHOD,
    n_components: int = 3,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    alpha: float = 0.7,
    s: int = 30,
    random_state: int = 42,
    save_to_mlflow: bool = False,
    mlflow_run_id: Optional[str] = None,
    mlflow_artifact_path: str = "visualizations"
) -> Optional[object]:
    """
    Visualize embeddings from multiple WAV files in 3D space.
    
    Each WAV file is processed to extract embeddings from multiple chunks,
    and all embeddings are visualized together with each file in a different color.
    
    Args:
        model_path: Path to model (local path, MLflow URI, or MLflow run_id)
        wav_files: List of paths to WAV files to visualize
        labels: Optional list of labels for each file (default: filename)
        device: Device to use (None = auto-detect)
        sample_rate: Sample rate (default: 16000)
        duration_sec: Duration of chunks in seconds (default: 1.0)
        chunks_per_file: Number of chunks to extract per file (default: 50)
        rms_threshold: RMS threshold for silence filtering (default: 0.005)
        embedding_dim: Embedding dimension (default: 64)
        output_file: Path to save visualization (None = don't save)
        reduction_method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of dimensions for visualization (2 or 3, default: 3)
        show_plot: Whether to display the plot (in Jupyter widget mode, set to False and return fig)
        figsize: Figure size (width, height)
        alpha: Transparency of points (0.0 to 1.0, default: 0.7)
        s: Size of points (default: 30)
        random_state: Random state for reproducibility
    
    Returns:
        Figure object if successful, None otherwise
    
    Example:
        # Visualize embeddings from two WAV files
        visualize_wav_files(
            model_path="runs:/<run_id>/models/model.pth",
            wav_files=["normal.wav", "anomaly.wav"],
            labels=["Normal", "Anomaly"],
            chunks_per_file=50
        )
    """
    import random
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Check dependencies
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install it with: pip install matplotlib")
    
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for dimensionality reduction. Install it with: pip install scikit-learn")
    
    # Try to import torchaudio and soundfile
    try:
        import torchaudio
        TORCHAUDIO_AVAILABLE = True
    except ImportError:
        TORCHAUDIO_AVAILABLE = False
    
    try:
        import soundfile as sf
        SOUNDFILE_AVAILABLE = True
    except ImportError:
        SOUNDFILE_AVAILABLE = False
    
    if not TORCHAUDIO_AVAILABLE and not SOUNDFILE_AVAILABLE:
        logger.error("torchaudio or soundfile is required for WAV file processing")
        return None
    
    # Determine device
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    logger.info("=" * 60)
    logger.info("3D Visualization of WAV Files")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"WAV files: {len(wav_files)}")
    logger.info(f"Device: {device}")
    
    # Import model loading function
    try:
        from ml.validate import load_model_mlflow, get_embedding
    except ImportError:
        from validate import load_model_mlflow, get_embedding
    
    # Import dataset class for transforms
    try:
        from ml.triplet_memory_dataset import TripletMemoryDataset
    except ImportError:
        from triplet_memory_dataset import TripletMemoryDataset
    
    # Import WAV loading function
    try:
        from ml.prepare_dataset import load_and_process_wav_file
    except ImportError:
        from prepare_dataset import load_and_process_wav_file
    
    # Load model
    logger.info("\n1. Loading model...")
    model = load_model_mlflow(model_path, device, embedding_dim=embedding_dim)
    model = model.to(device)
    model.eval()
    
    # Create transforms
    transform, amplitude_to_log = TripletMemoryDataset.create_mel_transform(sample_rate)
    target_len = int(sample_rate * duration_sec)
    
    # Generate labels if not provided
    if labels is None:
        labels = [os.path.basename(f) for f in wav_files]
    elif len(labels) != len(wav_files):
        raise ValueError(f"Number of labels ({len(labels)}) must match number of files ({len(wav_files)})")
    
    # Process each WAV file
    batches = []
    logger.info("\n2. Processing WAV files...")
    
    for i, wav_path in enumerate(wav_files):
        logger.info(f"\n   Processing file {i+1}/{len(wav_files)}: {os.path.basename(wav_path)}")
        
        if not os.path.exists(wav_path):
            logger.error(f"File not found: {wav_path}")
            continue
        
        # Load WAV file
        try:
            wav_int16 = load_and_process_wav_file(wav_path, sample_rate=sample_rate)
            logger.info(f"   Loaded: {wav_int16.shape[1] / sample_rate:.2f} seconds")
        except Exception as e:
            logger.error(f"   Failed to load WAV file: {e}")
            continue
        
        # Extract random chunks
        def extract_random_chunks(wav_int16: torch.Tensor, num_chunks: int, min_rms: Optional[float] = None) -> List[torch.Tensor]:
            """Extract random chunks from waveform and convert to embeddings."""
            chunks = []
            attempts = 0
            max_attempts = num_chunks * 10
            
            while len(chunks) < num_chunks and attempts < max_attempts:
                attempts += 1
                
                # Random crop
                total_len = wav_int16.shape[1]
                if total_len > target_len:
                    start = random.randint(0, total_len - target_len)
                    crop_int16 = wav_int16[:, start : start + target_len]
                else:
                    padding = target_len - total_len
                    crop_int16 = F.pad(wav_int16, (0, padding))
                
                # Convert int16 -> float32 normalized
                wav_crop_float = crop_int16.float() / 32767.0
                
                # RMS filtering
                if min_rms is not None:
                    rms = torch.sqrt(torch.mean(wav_crop_float**2)).item()
                    if rms < min_rms:
                        continue
                
                # Convert to mel spectrogram
                spec = transform(wav_crop_float)
                spec = amplitude_to_log(spec)
                
                # Get embedding
                with torch.no_grad():
                    vec = get_embedding(model, spec.unsqueeze(0), device)
                    chunks.append(vec.cpu().numpy()[0])
            
            return chunks
        
        # Extract embeddings
        embeddings_list = extract_random_chunks(
            wav_int16,
            chunks_per_file,
            min_rms=rms_threshold if rms_threshold > 0 else None
        )
        
        if len(embeddings_list) == 0:
            logger.warning(f"   No valid chunks extracted from {os.path.basename(wav_path)}")
            continue
        
        embeddings_array = np.array(embeddings_list)
        batches.append(embeddings_array)
        logger.info(f"   Extracted {len(embeddings_list)} embeddings")
    
    if len(batches) == 0:
        logger.error("No embeddings extracted from any file")
        return None
    
    # Extract MLflow run_id from model_path if it's an MLflow URI and not provided
    run_id_to_use = mlflow_run_id
    if run_id_to_use is None and save_to_mlflow:
        if model_path.startswith("runs:/"):
            # Extract run_id from URI: runs:/<run_id>/models/...
            parts = model_path.split('/')
            if len(parts) >= 2:
                run_id_to_use = parts[1]
                logger.info(f"Extracted MLflow run_id from model_path: {run_id_to_use}")
        elif len(model_path) == 32 and all(c in '0123456789abcdef' for c in model_path.lower()):
            # Model path is just a run_id
            run_id_to_use = model_path
            logger.info(f"Using model_path as MLflow run_id: {run_id_to_use}")
    
    # Visualize using visualize_batches
    logger.info("\n3. Visualizing embeddings...")
    try:
        # For widget mode in Jupyter, don't show plot here - let the returned figure handle it
        fig = visualize_batches(
            batches=batches,
            labels=labels,
            output_file=output_file,
            reduction_method=reduction_method,
            n_components=n_components,
            show_plot=show_plot,
            figsize=figsize,
            alpha=alpha,
            s=s,
            random_state=random_state,
            save_to_mlflow=save_to_mlflow,
            mlflow_run_id=run_id_to_use,
            mlflow_artifact_path=mlflow_artifact_path
        )
        logger.info("✅ Visualization complete!")
        return fig
    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)
        return None


def main():
    """Main function for standalone script execution."""
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib is required for visualization. Install it with: pip install matplotlib")
        return 1
    
    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required for dimensionality reduction. Install it with: pip install scikit-learn")
        return 1
    
    parser = argparse.ArgumentParser(
        description='Visualize batches of embedding vectors or WAV files in 3D space',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize embeddings from .npy files
  python visualize_embeddings.py \\
      --batches embeddings1.npy embeddings2.npy embeddings3.npy \\
      --labels "Sample 1" "Sample 2" "Sample 3" \\
      --output comparison.png

  # Visualize embeddings from WAV files
  python visualize_embeddings.py \\
      --model runs:/<run_id>/models/model.pth \\
      --wav-files normal.wav anomaly.wav \\
      --labels "Normal" "Anomaly" \\
      --chunks-per-file 50 \\
      --output wav_comparison.png

  # Use t-SNE instead of PCA
  python visualize_embeddings.py \\
      --batches embeddings1.npy embeddings2.npy \\
      --reduction-method tsne \\
      --output comparison_tsne.png
        """
    )
    
    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Visualization mode')
    
    # Mode 1: Visualize batches from .npy files
    parser_batches = subparsers.add_parser('batches', help='Visualize batches from .npy files')
    parser_batches.add_argument(
        '--batches', '-b',
        type=str,
        nargs='+',
        required=True,
        help='Paths to .npy files containing embedding batches (each file: [N, embedding_dim])'
    )
    parser_batches.add_argument(
        '--labels', '-l',
        type=str,
        nargs='+',
        default=None,
        help='Labels for each batch (default: auto-generated)'
    )
    parser_batches.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save visualization (default: show only, don\'t save)'
    )
    parser_batches.add_argument(
        '--reduction-method',
        type=str,
        choices=['pca', 'tsne'],
        default=DEFAULT_REDUCTION_METHOD,
        help=f'Dimensionality reduction method (default: {DEFAULT_REDUCTION_METHOD})'
    )
    parser_batches.add_argument(
        '--n-components',
        type=int,
        choices=[2, 3],
        default=3,
        help='Number of dimensions for visualization (2 or 3, default: 3)'
    )
    parser_batches.add_argument(
        '--no-show',
        action='store_true',
        help='Don\'t display the plot (useful when saving to file)'
    )
    parser_batches.add_argument(
        '--alpha',
        type=float,
        default=0.6,
        help='Transparency of points (0.0 to 1.0, default: 0.6)'
    )
    parser_batches.add_argument(
        '--point-size',
        type=int,
        default=20,
        help='Size of points (default: 20)'
    )
    
    # Mode 2: Visualize WAV files
    parser_wav = subparsers.add_parser('wav', help='Visualize embeddings from WAV files')
    parser_wav.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to model (local file or MLflow URI like runs:/<run_id>/models/model.pth)'
    )
    parser_wav.add_argument(
        '--wav-files', '-w',
        type=str,
        nargs='+',
        required=True,
        help='Paths to WAV files to visualize'
    )
    parser_wav.add_argument(
        '--labels', '-l',
        type=str,
        nargs='+',
        default=None,
        help='Labels for each WAV file (default: filename)'
    )
    parser_wav.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save visualization (default: show only, don\'t save)'
    )
    parser_wav.add_argument(
        '--chunks-per-file',
        type=int,
        default=50,
        help='Number of chunks to extract per file (default: 50)'
    )
    parser_wav.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Sample rate (default: 16000)'
    )
    parser_wav.add_argument(
        '--duration-sec',
        type=float,
        default=1.0,
        help='Duration of chunks in seconds (default: 1.0)'
    )
    parser_wav.add_argument(
        '--rms-threshold',
        type=float,
        default=0.005,
        help='RMS threshold for silence filtering (default: 0.005)'
    )
    parser_wav.add_argument(
        '--embedding-dim',
        type=int,
        default=64,
        help='Embedding dimension (default: 64)'
    )
    parser_wav.add_argument(
        '--reduction-method',
        type=str,
        choices=['pca', 'tsne'],
        default=DEFAULT_REDUCTION_METHOD,
        help=f'Dimensionality reduction method (default: {DEFAULT_REDUCTION_METHOD})'
    )
    parser_wav.add_argument(
        '--n-components',
        type=int,
        choices=[2, 3],
        default=3,
        help='Number of dimensions for visualization (2 or 3, default: 3)'
    )
    parser_wav.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'mps'],
        default=None,
        help='Device to use (default: auto-detect)'
    )
    parser_wav.add_argument(
        '--no-show',
        action='store_true',
        help='Don\'t display the plot (useful when saving to file)'
    )
    
    args = parser.parse_args()
    
    # Handle WAV mode
    if args.mode == 'wav' or (hasattr(args, 'model') and hasattr(args, 'wav_files') and args.model and args.wav_files):
        # Determine device
        if hasattr(args, 'device') and args.device:
            device = torch.device(args.device)
        else:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        
        try:
            fig = visualize_wav_files(
                model_path=args.model,
                wav_files=args.wav_files,
                labels=args.labels if hasattr(args, 'labels') else None,
                device=device,
                sample_rate=args.sample_rate if hasattr(args, 'sample_rate') else 16000,
                duration_sec=args.duration_sec if hasattr(args, 'duration_sec') else 1.0,
                chunks_per_file=args.chunks_per_file if hasattr(args, 'chunks_per_file') else 50,
                rms_threshold=args.rms_threshold if hasattr(args, 'rms_threshold') else 0.005,
                embedding_dim=args.embedding_dim if hasattr(args, 'embedding_dim') else 64,
                output_file=args.output if hasattr(args, 'output') else None,
                reduction_method=args.reduction_method if hasattr(args, 'reduction_method') else DEFAULT_REDUCTION_METHOD,
                n_components=args.n_components if hasattr(args, 'n_components') else 3,
                show_plot=not (hasattr(args, 'no_show') and args.no_show)
            )
            return 0 if fig is not None else 1
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return 1
    
    # Handle batches mode (default or explicit)
    elif args.mode == 'batches' or (hasattr(args, 'batches') and args.batches):
        batches_arg = args.batches
        
        # Load batches from files
        batches = []
        for batch_file in batches_arg:
            if not os.path.exists(batch_file):
                logger.error(f"File not found: {batch_file}")
                return 1
            
            try:
                batch = np.load(batch_file)
                batches.append(batch)
                logger.info(f"Loaded {batch_file}: shape {batch.shape}")
            except Exception as e:
                logger.error(f"Error loading {batch_file}: {e}")
                return 1
        
        try:
            visualize_batches(
                batches=batches,
                labels=args.labels if hasattr(args, 'labels') else None,
                output_file=args.output if hasattr(args, 'output') else None,
                reduction_method=args.reduction_method if hasattr(args, 'reduction_method') else DEFAULT_REDUCTION_METHOD,
                n_components=args.n_components if hasattr(args, 'n_components') else 3,
                show_plot=not (hasattr(args, 'no_show') and args.no_show),
                alpha=args.alpha if hasattr(args, 'alpha') else 0.6,
                s=args.point_size if hasattr(args, 'point_size') else 20
            )
            return 0
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return 1
    
    else:
        parser.print_help()
        logger.error("Please specify mode: 'batches' (for .npy files) or 'wav' (for WAV files)")
        return 1


if __name__ == "__main__":
    exit(main())
