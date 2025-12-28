#!/usr/bin/env python3
"""
Early Stopping Mechanism for Training

This module provides an early stopping callback for training loops with various criteria.
It can stop training based on patience (no improvement), minimum loss threshold, or both.

It can be run as a standalone script for testing or imported as a module for training.

Configuration:
    The module supports configuration via environment variables or .env file.
    See env.example for available configuration options.
    If python-dotenv is installed, .env file will be automatically loaded.

Usage as module:
    from early_stopping import EarlyStopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    for epoch in range(epochs):
        loss = train_epoch(...)
        if early_stopping(loss, epoch, model):
            break
    early_stopping.restore_weights(model)

Usage as script:
    python early_stopping.py [--test]
"""

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
DEFAULT_PATIENCE = int(os.getenv('EARLY_STOPPING_PATIENCE', '10'))
DEFAULT_MIN_DELTA = float(os.getenv('EARLY_STOPPING_MIN_DELTA', '0.0'))
DEFAULT_MIN_LOSS = os.getenv('EARLY_STOPPING_MIN_LOSS', None)
DEFAULT_MIN_LOSS = float(DEFAULT_MIN_LOSS) if DEFAULT_MIN_LOSS else None
DEFAULT_MODE = os.getenv('EARLY_STOPPING_MODE', 'min')
DEFAULT_VERBOSE = os.getenv('EARLY_STOPPING_VERBOSE', 'true').lower() == 'true'
DEFAULT_RESTORE_BEST_WEIGHTS = os.getenv('EARLY_STOPPING_RESTORE_BEST_WEIGHTS', 'true').lower() == 'true'


class EarlyStopping:
    """
    Early stopping mechanism for training with various criteria.
    
    This class monitors training metrics and stops training when:
    - No improvement is seen for 'patience' epochs (patience-based stopping)
    - A minimum loss threshold is reached (threshold-based stopping)
    
    It can also restore the best model weights when training stops.
    
    Args:
        patience: Number of epochs without improvement before stopping (default: 10)
        min_delta: Minimum change to qualify as an improvement (default: 0.0)
        min_loss: Minimum loss threshold to stop training (default: None)
        mode: 'min' for minimizing loss, 'max' for maximizing metric (default: 'min')
        verbose: Print early stopping information (default: True)
        restore_best_weights: Restore best weights when stopping (default: True)
    """
    
    def __init__(self, patience: int = DEFAULT_PATIENCE, min_delta: float = DEFAULT_MIN_DELTA,
                 min_loss: float = None, mode: str = DEFAULT_MODE, verbose: bool = DEFAULT_VERBOSE,
                 restore_best_weights: bool = DEFAULT_RESTORE_BEST_WEIGHTS):
        self.patience = patience
        self.min_delta = min_delta
        self.min_loss = min_loss if min_loss is not None else DEFAULT_MIN_LOSS
        self.mode = mode
        self.verbose = verbose
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.best_weights = None
        
        if self.verbose:
            logger.info(f"EarlyStopping initialized: patience={patience}, min_delta={min_delta}, "
                       f"min_loss={self.min_loss}, mode={mode}")
        
    def __call__(self, score: float, epoch: int, model=None) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            score: Current metric value (loss or other metric)
            epoch: Current epoch number (0-indexed)
            model: Model to save best weights (optional)
        
        Returns:
            bool: True if training should be stopped
        """
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.best_epoch = epoch
            if model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.verbose:
                logger.info(f"Initial score: {score:.6f} (epoch {epoch + 1})")
        else:
            # Determine if there's improvement
            if self.mode == 'min':
                is_better = score < (self.best_score - self.min_delta)
            else:  # mode == 'max'
                is_better = score > (self.best_score + self.min_delta)
            
            if is_better:
                # Improvement found
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
                if model is not None:
                    self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                if self.verbose:
                    logger.info(f"✅ Improvement! Best score: {self.best_score:.6f} (epoch {epoch + 1})")
            else:
                # No improvement
                self.counter += 1
                if self.verbose:
                    logger.info(f"⏳ No improvement {self.counter}/{self.patience}. "
                              f"Best: {self.best_score:.6f} (epoch {self.best_epoch + 1})")
        
        # Check minimum loss threshold
        if self.min_loss is not None:
            if self.mode == 'min' and score <= self.min_loss:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"🛑 Reached minimum loss threshold: {score:.6f} <= {self.min_loss:.6f}")
                return True
        
        # Check patience
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                logger.warning(f"🛑 Early stopping! Patience: {self.patience} epochs without improvement.")
                logger.info(f"   Best result: {self.best_score:.6f} at epoch {self.best_epoch + 1}")
            return True
        
        return False
    
    def restore_weights(self, model):
        """
        Restore the best model weights.
        
        Args:
            model: Model to restore weights to
        """
        if self.best_weights is not None and self.restore_best_weights:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                logger.info(f"✅ Restored best weights from epoch {self.best_epoch + 1}")
        elif self.best_weights is None:
            logger.warning("No best weights to restore. Model was not provided during __call__.")
        elif not self.restore_best_weights:
            logger.info("Weight restoration disabled.")
    
    def get_info(self) -> dict:
        """
        Get information about early stopping state.
        
        Returns:
            Dictionary with early stopping state information
        """
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'counter': self.counter,
            'patience': self.patience,
            'early_stopped': self.early_stop,
            'min_loss': self.min_loss,
            'mode': self.mode
        }
    
    def reset(self):
        """
        Reset early stopping state (useful for multiple training runs).
        """
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        self.best_weights = None
        if self.verbose:
            logger.info("Early stopping state reset")


def test_early_stopping():
    """
    Test early stopping with simulated training loss values.
    
    Returns:
        True if all tests pass, False otherwise
    """
    logger.info("Testing EarlyStopping...")
    
    try:
        # Test 1: Patience-based stopping
        logger.info("\nTest 1: Patience-based stopping")
        early_stopping = EarlyStopping(patience=3, min_delta=0.001, verbose=True)
        
        # Simulate decreasing loss
        # Epoch 0: 1.0 (initial)
        # Epoch 1: 0.9 (improvement)
        # Epoch 2: 0.85 (improvement)
        # Epoch 3: 0.84 (improvement - best)
        # Epoch 4-6: 0.84 (no improvement, patience exhausted)
        losses = [1.0, 0.9, 0.85, 0.84, 0.84, 0.84, 0.84]
        stopped = False
        
        for epoch, loss in enumerate(losses):
            if early_stopping(loss, epoch):
                stopped = True
                logger.info(f"Stopped at epoch {epoch + 1}")
                break
        
        if not stopped:
            logger.error("Test 1 failed: Should have stopped due to patience")
            return False
        
        info = early_stopping.get_info()
        # Best was at epoch 3 (loss=0.84), which is better than epoch 2 (loss=0.85)
        if info['best_epoch'] != 3:
            logger.error(f"Test 1 failed: Best epoch should be 3 (loss=0.84), got {info['best_epoch']}")
            return False
        
        if abs(info['best_score'] - 0.84) > 0.001:
            logger.error(f"Test 1 failed: Best score should be 0.84, got {info['best_score']}")
            return False
        
        logger.info("✅ Test 1 passed")
        
        # Test 2: Minimum loss threshold
        logger.info("\nTest 2: Minimum loss threshold")
        early_stopping2 = EarlyStopping(patience=10, min_loss=0.5, verbose=True)
        
        losses2 = [1.0, 0.9, 0.8, 0.6, 0.5]
        stopped2 = False
        
        for epoch, loss in enumerate(losses2):
            if early_stopping2(loss, epoch):
                stopped2 = True
                logger.info(f"Stopped at epoch {epoch + 1} due to min_loss threshold")
                break
        
        if not stopped2:
            logger.error("Test 2 failed: Should have stopped due to min_loss threshold")
            return False
        
        logger.info("✅ Test 2 passed")
        
        # Test 3: No stopping (continuous improvement)
        logger.info("\nTest 3: No stopping (continuous improvement)")
        early_stopping3 = EarlyStopping(patience=3, min_delta=0.001, verbose=False)
        
        losses3 = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        stopped3 = False
        
        for epoch, loss in enumerate(losses3):
            if early_stopping3(loss, epoch):
                stopped3 = True
                break
        
        if stopped3:
            logger.error("Test 3 failed: Should not have stopped with continuous improvement")
            return False
        
        logger.info("✅ Test 3 passed")
        
        logger.info("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False


def main():
    """Main function for standalone script execution."""
    parser = argparse.ArgumentParser(
        description='Test EarlyStopping mechanism'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Run early stopping tests'
    )
    parser.add_argument(
        '--patience', '-p',
        type=int,
        default=DEFAULT_PATIENCE,
        help=f'Patience value (default: {DEFAULT_PATIENCE})'
    )
    parser.add_argument(
        '--min-delta',
        type=float,
        default=DEFAULT_MIN_DELTA,
        help=f'Minimum delta for improvement (default: {DEFAULT_MIN_DELTA})'
    )
    parser.add_argument(
        '--min-loss',
        type=float,
        default=None,
        help='Minimum loss threshold (default: None)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['min', 'max'],
        default=DEFAULT_MODE,
        help=f'Mode: min or max (default: {DEFAULT_MODE})'
    )
    
    args = parser.parse_args()
    
    try:
        if args.test:
            success = test_early_stopping()
            return 0 if success else 1
        
        # Default: show configuration
        logger.info("EarlyStopping Configuration:")
        logger.info("=" * 60)
        logger.info(f"Patience: {args.patience}")
        logger.info(f"Min Delta: {args.min_delta}")
        logger.info(f"Min Loss: {args.min_loss if args.min_loss else 'None'}")
        logger.info(f"Mode: {args.mode}")
        logger.info("=" * 60)
        logger.info("\nUse --test to run tests")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
