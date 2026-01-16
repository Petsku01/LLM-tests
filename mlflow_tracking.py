"""MLflow experiment tracking."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

class MLflowExperimentTracker:
    """Log experiments to MLflow."""
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        auto_log: bool = False
    ):
        self.experiment_name = experiment_name
        self.tags = tags or {}
        self.auto_log = auto_log
        self.run = None
        # Get the process rank from distributed launcher
        self._rank = int(os.environ.get('RANK', 0))
        
        # Only rank 0 should initialize MLflow
        # All other ranks just become no-ops for logging
        if self._rank == 0:
            try:
                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}")
    
    def start_run(self, run_name: Optional[str] = None) -> Optional[Any]:
        """Start a new run. Only works on rank 0 in distributed training."""
        if self._rank != 0:
            return None
        
        try:
            # End any existing run first (cleanup from previous run)
            if mlflow.active_run():
                mlflow.end_run()
            
            self.run = mlflow.start_run(run_name=run_name)
            
            # Add any tags we were given
            for key, value in self.tags.items():
                mlflow.set_tag(key, str(value))
            
            logger.info(f"MLflow run started: {run_name or 'unnamed'}")
            return self.run
            
        except Exception as e:
            # Connection error? Server down? Log it but don't crash
            logger.error(f"MLflow failed to start run: {e}")
            # Return None so calling code knows it failed
            return None
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters, filtering out stuff that doesn't serialize."""
        if self._rank != 0:
            return
        
        try:
            clean_params = {}
            for key, value in params.items():
                if value is None:
                    continue
                if isinstance(value, (int, float, str, bool)):
                    clean_params[key] = value
                elif isinstance(value, (list, tuple)) and len(value) < 50:
                    clean_params[key] = str(value)
            
            if clean_params:
                mlflow.log_params(clean_params)
        except Exception as e:
            logger.warning(f"Could not log params: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        if self._rank != 0:
            return
        
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Could not log metrics: {e}")
    
    def log_model_info(self, model: torch.nn.Module):
        """Log basic model info (parameter counts)."""
        if self._rank != 0:
            return
        
        try:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mlflow.log_metrics({
                'total_parameters': float(total),
                'trainable_parameters': float(trainable),
            })
        except Exception as e:
            logger.warning(f"Could not log model info: {e}")
    
    def end_run(self):
        """End the MLflow run."""
        if self._rank == 0:
            try:
                mlflow.end_run()
            except Exception:
                pass


class MLflowCallback(TrainerCallback):
    """Logs training metrics to MLflow as training progresses."""
    
    def __init__(self, tracker: MLflowExperimentTracker):
        self.tracker = tracker
    
    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Log training start."""
        if not self.tracker.run:
            self.tracker.start_run()
    
    
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control,
        logs: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Log metrics during training."""
        if logs and self.tracker._rank == 0:
            try:
                step = state.global_step
                self.tracker.log_metrics(logs, step=step)
            except Exception as e:
                logger.warning(f"Failed to log metrics: {e}")


def create_experiment_tracker(
    experiment_name: str,
    model_name: str = "model",
    dataset_name: str = "dataset",
    tracking_uri: Optional[str] = None
) -> MLflowExperimentTracker:
    """Factory to create an experiment tracker."""
    tags = {
        "model": model_name,
        "dataset": dataset_name,
    }
    
    return MLflowExperimentTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        tags=tags,
        auto_log=False  # Avoid conflicts with manual logging
    )
