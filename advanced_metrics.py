"""Metrics for evaluating fine-tuned models."""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class AdvancedMetrics:
    """Calculate evaluation metrics for models."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    @staticmethod
    def perplexity(loss: float) -> float:
        """Convert loss to perplexity."""
        try:
            if loss < 0 or not math.isfinite(loss):
                return float('inf')
            return math.exp(loss)
        except OverflowError:
            return float('inf')
    
    @staticmethod
    def token_accuracy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100
    ) -> float:
        """
        Accuracy of token predictions.
        
        Args:
            logits: Model output shape [batch, seq_len, vocab_size]
            labels: Ground truth shape [batch, seq_len]
            ignore_index: Tokens with this index are ignored (usually padding)
        
        Returns:
            Fraction of correctly predicted tokens
        """
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(logits)}")
        if not isinstance(labels, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(labels)}")
        
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        
        # Create mask for valid tokens
        mask = labels != ignore_index
        
        # Count correct predictions
        if mask.sum() == 0:
            logger.warning("All tokens ignored in accuracy calculation")
            return 0.0
        
        correct = (preds == labels) & mask
        return correct.sum().item() / mask.sum().item()
    
    @staticmethod
    def top_k_accuracy(
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5,
        ignore_index: int = -100
    ) -> float:
        """
        Top-k accuracy (prediction is correct if true label in top-k).
        
        Args:
            logits: Model output [batch, seq_len, vocab_size]
            labels: Ground truth [batch, seq_len]
            k: Check if true label in top-k
            ignore_index: Ignore tokens with this index
        
        Returns:
            Top-k accuracy
        """
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected Tensor, got {type(logits)}")
        
        # Handle edge case
        if k > logits.shape[-1]:
            logger.warning(f"k={k} larger than vocab size {logits.shape[-1]}, using vocab size")
            k = logits.shape[-1]
        
        # Get top-k predictions
        _, top_k = torch.topk(logits, k, dim=-1)
        
        # Expand labels to match
        labels_exp = labels.unsqueeze(-1).expand_as(top_k)
        
        # Check if true label in top-k
        mask = labels != ignore_index
        if mask.sum() == 0:
            return 0.0
        
        correct = (top_k == labels_exp).any(dim=-1) & mask
        return correct.sum().item() / mask.sum().item()
    
    @staticmethod
    def token_f1(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
        average: str = 'weighted'
    ) -> Dict[str, float]:
        """
        F1, precision, and recall for token-level classification.
        
        Args:
            logits: Model output [batch, seq_len, vocab_size]
            labels: Ground truth [batch, seq_len]
            ignore_index: Ignore tokens with this index
            average: 'weighted', 'macro', or 'micro'
        
        Returns:
            Dict with f1, precision, recall
        """
        preds = torch.argmax(logits, dim=-1)
        
        # Flatten and filter
        mask = labels != ignore_index
        preds_flat = preds[mask].cpu().numpy()
        labels_flat = labels[mask].cpu().numpy()
        
        if len(labels_flat) == 0:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        return {
            'f1': float(f1_score(labels_flat, preds_flat, average=average, zero_division=0)),
            'precision': float(precision_score(labels_flat, preds_flat, average=average, zero_division=0)),
            'recall': float(recall_score(labels_flat, preds_flat, average=average, zero_division=0))
        }
    
    @staticmethod
    def entropy(logits: torch.Tensor, temperature: float = 1.0) -> Dict[str, float]:
        """
        Calculate entropy of predictions.
        
        High entropy = uncertain, low entropy = confident.
        
        Args:
            logits: Model output [batch, seq_len, vocab_size]
            temperature: Scale logits (higher = more uniform)
        
        Returns:
            Dict with mean, min, max entropy
        """
        # Apply temperature
        logits_scaled = logits / temperature
        
        # Get probabilities
        probs = torch.softmax(logits_scaled, dim=-1)
        
        # Calculate entropy: -sum(p * log(p))
        log_probs = torch.log_softmax(logits_scaled, dim=-1)
        entropy_vals = -(probs * log_probs).sum(dim=-1)
        
        # Convert to numpy for stats
        ent_np = entropy_vals.detach().cpu().numpy()
        
        return {
            'mean': float(np.mean(ent_np)),
            'std': float(np.std(ent_np)),
            'min': float(np.min(ent_np)),
            'max': float(np.max(ent_np))
        }
    
    @staticmethod
    def calibration(
        logits: torch.Tensor,
        labels: torch.Tensor,
        n_bins: int = 10,
        ignore_index: int = -100
    ) -> Dict[str, float]:
        """
        Expected Calibration Error (ECE).
        
        Measures if model confidence matches accuracy.
        Low ECE = model is well-calibrated.
        
        Args:
            logits: Model output
            labels: Ground truth
            n_bins: Number of confidence bins
            ignore_index: Ignore tokens with this index
        
        Returns:
            Dict with ECE and related metrics
        """
        preds = torch.argmax(logits, dim=-1)
        confs = torch.max(torch.softmax(logits, dim=-1), dim=-1)[0]
        
        # Filter
        mask = labels != ignore_index
        preds_f = preds[mask].cpu()
        labels_f = labels[mask].cpu()
        confs_f = confs[mask].cpu()
        
        if len(preds_f) == 0:
            return {'ece': 0.0, 'mce': 0.0}
        
        # Bin predictions
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ece = 0.0
        mce = 0.0
        
        for i in range(n_bins):
            mask_bin = (confs_f >= bin_edges[i]) & (confs_f < bin_edges[i+1])
            
            if mask_bin.sum() == 0:
                continue
            
            conf_bin = confs_f[mask_bin].numpy()
            acc_bin = (preds_f[mask_bin] == labels_f[mask_bin]).numpy().astype(float)
            
            mean_conf = np.mean(conf_bin)
            mean_acc = np.mean(acc_bin)
            
            ece += mask_bin.sum() / len(preds_f) * abs(mean_conf - mean_acc)
            mce = max(mce, abs(mean_conf - mean_acc))
        
        return {
            'ece': float(ece),
            'mce': float(mce)
        }


class GradientAnalyzer:
    """Analyzes gradient flow during training."""
    
    @staticmethod
    def gradient_norm(model: torch.nn.Module, norm_type: float = 2.0) -> Dict[str, float]:
        """
        Calculate gradient norms per layer.
        
        Args:
            model: Model to analyze
            norm_type: 1.0 for L1, 2.0 for L2, etc
        
        Returns:
            Dict with global norm and per-layer info
        """
        total_norm = 0.0
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad, p=norm_type).item()
                total_norm += grad_norm ** norm_type
                layer_norms[name] = grad_norm
        
        total_norm = total_norm ** (1.0 / norm_type)
        
        return {
            'total': total_norm,
            'mean': float(np.mean([v for v in layer_norms.values()]) if layer_norms else 0.0),
            'max': float(max(layer_norms.values()) if layer_norms else 0.0),
            'min': float(min(layer_norms.values()) if layer_norms else 0.0)
        }
    
    @staticmethod
    def check_gradient_issues(model: torch.nn.Module, verbose: bool = False) -> Dict[str, bool]:
        """
        Check for vanishing or exploding gradients.
        
        Args:
            model: Model to check
            verbose: Print warnings
        
        Returns:
            Dict with issues found
        """
        issues = {
            'vanishing': False,
            'exploding': False,
            'nan': False,
            'inf': False
        }
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Check for NaN/Inf
            if torch.isnan(grad).any():
                issues['nan'] = True
                if verbose:
                    logger.warning(f"NaN gradient in {name}")
            
            if torch.isinf(grad).any():
                issues['inf'] = True
                if verbose:
                    logger.warning(f"Inf gradient in {name}")
            
            # Check magnitude
            grad_norm = torch.norm(grad).item()
            
            if grad_norm < 1e-7:
                issues['vanishing'] = True
                if verbose:
                    logger.warning(f"Tiny gradient in {name}: {grad_norm:.2e}")
            
            if grad_norm > 100.0:
                issues['exploding'] = True
                if verbose:
                    logger.warning(f"Large gradient in {name}: {grad_norm:.2e}")
        
        return issues


def evaluate_on_dataset(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Quick evaluation on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: Data to evaluate on
        device: Where to run
        max_batches: Only evaluate on this many batches (for speed)
    
    Returns:
        Dict with accuracy, f1, loss, etc
    """
    model.eval()
    metrics_obj = AdvancedMetrics(device)
    
    total_loss = 0.0
    total_acc = 0.0
    total_f1_results = {'f1': [], 'precision': [], 'recall': []}
    num_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            # Move to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = batch.to(device)
            
            # Forward pass
            outputs = model(**batch) if isinstance(batch, dict) else model(batch)
            
            # Get loss and logits
            loss = outputs.loss if hasattr(outputs, 'loss') else None
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            if loss is not None:
                total_loss += loss.item()
            
            # Get labels
            labels = batch['labels'] if isinstance(batch, dict) else batch
            
            # Calculate metrics
            acc = metrics_obj.token_accuracy(logits, labels)
            f1_dict = metrics_obj.token_f1(logits, labels)
            
            total_acc += acc
            for k, v in f1_dict.items():
                total_f1_results[k].append(v)
            
            num_batches += 1
    
    if num_batches == 0:
        return {}
    
    return {
        'accuracy': total_acc / num_batches,
        'f1': float(np.mean(total_f1_results['f1'])) if total_f1_results['f1'] else 0.0,
        'precision': float(np.mean(total_f1_results['precision'])) if total_f1_results['precision'] else 0.0,
        'recall': float(np.mean(total_f1_results['recall'])) if total_f1_results['recall'] else 0.0,
        'loss': total_loss / num_batches if total_loss > 0 else 0.0,
        'perplexity': metrics_obj.perplexity(total_loss / num_batches) if total_loss > 0 else 0.0
    }
