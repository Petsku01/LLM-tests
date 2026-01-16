"""Learning rate finder using Leslie Smith's method."""

import logging
import math
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class LearningRateFinder:
    """Find learning rates using Leslie Smith's method."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: str = "cuda"
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        
        try:
            model_device = next(model.parameters()).device
            if str(model_device) != device:
                logger.warning(
                    f"Model on {model_device} but finder expects {device}"
                )
        except StopIteration:
            logger.warning("Model has no parameters")
        
        # Store checkpoint for restoration
        self.checkpoint_saved = False
        self.best_loss = float('inf')
        self.history = {'lr': [], 'loss': [], 'smooth_loss': []}
        self.best_lr = None
    
    def _save_checkpoint(self):
        """Save model state before starting test."""
        try:
            self.model_state = self.model.state_dict()
            self.opt_state = self.optimizer.state_dict()
            self.checkpoint_saved = True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _restore_checkpoint(self):
        """Restore model state after test."""
        if not self.checkpoint_saved:
            return
        
        try:
            self.model.load_state_dict(self.model_state)
            self.optimizer.load_state_dict(self.opt_state)
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
    
    def range_test(
        self,
        train_loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Run the LR range test.
        
        Gradually increases LR and records loss. When loss starts 
        exploding, we've found the upper limit.
        
        Args:
            train_loader: Data to train on
            start_lr: Start here
            end_lr: End here
            num_iter: Test this many batches
            smooth_f: Smoothing for loss values (0.05 is standard)
            diverge_th: Stop if loss > best_loss * this value (prevents runaway)
            verbose: Print progress
        
        Returns:
            History dict with lrs and losses
        
        Note: Divergence detection is conservative to handle noisy batches.
        """
        # Save model state first before we mess with it
        self._save_checkpoint()
        
        self.model.train()
        
        # Calculate how much to multiply LR each step
        lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        current_lr = start_lr
        self.optimizer.param_groups[0]['lr'] = current_lr
        
        # Track smoothed loss with bias correction
        avg_loss = 0.0
        batch_num = 0
        
        # Create iterator from dataloader
        iterator = iter(train_loader)
        
        if verbose:
            logger.info(f"LR range test: {start_lr:.2e} to {end_lr:.2e}, {num_iter} steps")
        
        for step in range(num_iter):
            # Get next batch (restart if needed)
            try:
                batch = next(iterator)
            except StopIteration:
                # Restart iterator if we run out
                iterator = iter(train_loader)
                try:
                    batch = next(iterator)
                except StopIteration:
                    logger.error("Dataset exhausted during LR finder")
                    break
            
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(self.device)
            
            # Forward pass
            try:
                self.optimizer.zero_grad()
                
                # Get loss from model
                if isinstance(batch, dict):
                    outputs = self.model(**batch)
                else:
                    outputs = self.model(batch)
                
                # Handle different output types
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                elif isinstance(outputs, torch.Tensor):
                    loss = outputs
                else:
                    raise ValueError(f"Unexpected model output: {type(outputs)}")
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error("OOM during LR test - try smaller batch or fewer iterations")
                    break
                raise
            
            # Smooth the loss
            batch_num += 1
            loss_val = loss.item()
            
            # Check for NaN/Inf
            if not (math.isfinite(loss_val)):
                logger.warning(f"Loss is {loss_val} at LR={current_lr:.2e}, stopping")
                break
            
            # Apply smoothing with bias correction
            avg_loss = smooth_f * loss_val + (1 - smooth_f) * avg_loss
            smoothed = avg_loss / (1 - (1 - smooth_f) ** batch_num)
            
            # Track best loss
            if smoothed < self.best_loss:
                self.best_loss = smoothed
            
            # Store history
            self.history['lr'].append(current_lr)
            self.history['loss'].append(loss_val)
            self.history['smooth_loss'].append(smoothed)
            
            # Check if loss diverged
            if smoothed > diverge_th * self.best_loss:
                logger.warning(f"Loss diverged at LR={current_lr:.2e}, stopping")
                break
            
            # Increase LR for next iteration
            current_lr *= lr_mult
            self.optimizer.param_groups[0]['lr'] = current_lr
            
            # Progress
            if verbose and (step + 1) % max(1, num_iter // 10) == 0:
                logger.info(f"  Step {step+1}/{num_iter}: LR={current_lr:.2e}, Loss={smoothed:.4f}")
        
        # Find the best learning rate
        self.best_lr = self._find_best_lr()
        
        # Restore original state
        self._restore_checkpoint()
        
        if verbose and self.best_lr:
            logger.info(f"Suggested LR: {self.best_lr:.2e}")
        
        return self.history
    
    def _find_best_lr(self) -> Optional[float]:
        """
        Find the best LR from the test results.
        
        Looks for the point with steepest descent (most negative gradient).
        Falls back to 1/10 of the minimum loss LR if we can't find steepest.
        """
        if len(self.history['smooth_loss']) < 10:
            # Too few data points
            if len(self.history['lr']) > 0:
                return self.history['lr'][len(self.history['lr']) // 2]
            return None
        
        losses = np.array(self.history['smooth_loss'])
        lrs = np.array(self.history['lr'])
        
        # Skip first few (usually noisy)
        skip = min(10, len(losses) // 10)
        if skip >= len(losses):
            skip = 0
        
        losses = losses[skip:]
        lrs = lrs[skip:]
        
        if len(losses) == 0:
            return None
        
        # Find minimum loss
        min_idx = np.argmin(losses)
        min_lr = lrs[min_idx]
        
        # Calculate gradient on log scale
        log_lrs = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)
        
        # Find steepest descent (most negative gradient) before minimum
        if min_idx > 0:
            steepest_idx = np.argmin(gradients[:min_idx])
            steepest_lr = lrs[steepest_idx]
        else:
            steepest_lr = lrs[0]
        
        # Use steepest descent OR 1/10 of min, whichever is smaller
        # (We want to be conservative to avoid divergence during actual training)
        return min(steepest_lr, min_lr / 10.0)
    
    def plot(
        self,
        save_path: Optional[str] = None,
        skip_start: int = 10,
        skip_end: int = 5
    ):
        """
        Plot the LR finder results.
        
        Args:
            save_path: Save to this file (if provided)
            skip_start: Ignore first N points (usually noisy)
            skip_end: Ignore last N points
        """
        if len(self.history['lr']) == 0:
            logger.warning("No data to plot. Run range_test() first.")
            return
        
        # Prepare data
        lrs = self.history['lr'][skip_start:]
        losses = self.history['smooth_loss'][skip_start:]
        
        if skip_end > 0:
            lrs = lrs[:-skip_end]
            losses = losses[:-skip_end]
        
        if len(lrs) == 0:
            logger.warning("Not enough data to plot after skipping")
            return
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: LR vs Loss
        ax1.plot(lrs, losses, 'b-', linewidth=1.5)
        ax1.set_xscale('log')
        ax1.set_xlabel('Learning Rate (log scale)', fontsize=11)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.set_title('Learning Rate Finder', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        if self.best_lr:
            ax1.axvline(x=self.best_lr, color='red', linestyle='--', 
                        linewidth=2, label=f'Best LR: {self.best_lr:.2e}')
            ax1.legend(fontsize=10)
        
        # Plot 2: Loss gradient
        log_lrs = np.log10(lrs)
        gradients = np.gradient(losses, log_lrs)
        ax2.plot(lrs, gradients, 'g-', linewidth=1.5)
        ax2.set_xscale('log')
        ax2.set_xlabel('Learning Rate (log scale)', fontsize=11)
        ax2.set_ylabel('Loss Gradient', fontsize=11)
        ax2.set_title('Loss Gradient (steeper = better training)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


class OneCycleLR(_LRScheduler):
    """
    Implements the 1cycle learning rate schedule.
    
    Increases LR from low to high, then decreases to very low.
    Found to enable training at higher effective learning rates.
    
    Reference: Smith & Topin (2019)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 100000.0,
        last_epoch: int = -1
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            max_lr: Maximum LR during training
            total_steps: Total number of training steps
            pct_start: Spend this % of steps increasing LR
            div_factor: Initial LR = max_lr / div_factor
            final_div_factor: Final LR = max_lr / final_div_factor
            last_epoch: Resume from this epoch (default: -1 means start)
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # Calculate start and end LRs
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate LR for current step."""
        step = self.last_epoch
        
        # If past total steps, use final LR
        if step >= self.total_steps:
            return [self.final_lr for _ in self.base_lrs]
        
        # Where are we in the cycle? (0 to 1)
        cycle_pct = step / self.total_steps
        
        if cycle_pct < self.pct_start:
            # First phase: increase from initial to max
            phase_pct = cycle_pct / self.pct_start
            # Use cosine curve for smooth increase
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * \
                 (1 - math.cos(math.pi * phase_pct)) / 2
        else:
            # Second phase: decrease from max to final
            phase_pct = (cycle_pct - self.pct_start) / (1 - self.pct_start)
            # Use cosine curve for smooth decrease
            lr = self.final_lr + (self.max_lr - self.final_lr) * \
                 (1 + math.cos(math.pi * phase_pct)) / 2
        
        return [lr for _ in self.base_lrs]


def find_lr(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    device: str = "cuda",
    plot_path: Optional[str] = None,
) -> float:
    """
    Convenience function to quickly find a good learning rate.
    
    Args:
        model: Model to train
        train_loader: Training data
        optimizer: Optimizer to use
        device: cuda or cpu
        plot_path: Where to save the plot
    
    Returns:
        Suggested learning rate
    """
    finder = LearningRateFinder(model, optimizer, device)
    finder.range_test(
        train_loader,
        start_lr=1e-7,
        end_lr=10.0,
        num_iter=min(100, len(train_loader))
    )
    
    if plot_path:
        finder.plot(save_path=plot_path)
    
    return finder.best_lr or 1e-4  # Fallback to 1e-4 if no good LR found
