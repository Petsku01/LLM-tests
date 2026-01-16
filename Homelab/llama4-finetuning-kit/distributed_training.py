"""DeepSpeed ZeRO configuration helpers."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DeepSpeedConfig:
    """Creates standard DeepSpeed config files."""
    
    @staticmethod
    def zero_stage_2(
        micro_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        gradient_clipping: float = 1.0,
        offload_optimizer: bool = False
    ) -> Dict[str, Any]:
        """
        ZeRO Stage 2: Shard optimizer states across GPUs.
        
        Memory savings: ~50% vs normal training
        
        Args:
            micro_batch_size: Batch size per GPU
            gradient_accumulation_steps: How many steps to accumulate
            gradient_clipping: Max gradient norm for stability
            offload_optimizer: Move optimizer to CPU (slower but saves memory)
        
        Returns:
            Config dict to pass to DeepSpeed
        
        Stage 3 can be unstable for larger models.
        """
        config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": gradient_clipping,
            
            "fp16": {
                "enabled": "auto",
            },
            
            "bf16": {
                "enabled": "auto"
            },
            
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu" if offload_optimizer else "none",
                    "pin_memory": offload_optimizer
                }
            }
        }
        
        return config
    
    @staticmethod
    def zero_stage_3(
        micro_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        gradient_clipping: float = 1.0,
        offload_optimizer: bool = True,
        offload_param: bool = True,
        sub_group_size: int = 1e9
    ) -> Dict[str, Any]:
        """
        ZeRO Stage 3: Shard optimizer + gradients + parameters.
        
        Memory savings: ~60-70% vs normal training
        
        More aggressive than Stage 2. Use Stage 2 if instability occurs.
        
        Args:
            micro_batch_size: Batch size per GPU
            gradient_accumulation_steps: Accumulate gradients
            gradient_clipping: Max gradient norm
            offload_optimizer: Move optimizer to CPU
            offload_param: Move params to CPU
            sub_group_size: Partition size for ZeRO
        
        Returns:
            Config dict
        """
        config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_clipping": gradient_clipping,
            
            "fp16": {
                "enabled": "auto",
            },
            
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu" if offload_optimizer else "none",
                    "pin_memory": offload_optimizer
                },
                "offload_param": {
                    "device": "cpu" if offload_param else "none",
                    "pin_memory": offload_param
                },
                "sub_group_size": int(sub_group_size)
            }
        }
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], path: str = "ds_config.json"):
        """
        Save config to JSON file.
        
        Args:
            config: Config dict from zero_stage_2() or zero_stage_3()
            path: Where to save
        """
        try:
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved DeepSpeed config to {path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        Basic sanity checks on config.
        
        Args:
            config: Config dict to validate
        
        Returns:
            True if looks reasonable
        """
        # Check for required keys
        required = ['train_micro_batch_size_per_gpu', 'zero_optimization']
        for key in required:
            if key not in config:
                logger.warning(f"Missing required key: {key}")
                return False
        
        # Check stage
        stage = config['zero_optimization'].get('stage')
        if stage not in [1, 2, 3]:
            logger.warning(f"Invalid ZeRO stage: {stage}")
            return False
        
        # Check batch size
        batch_size = config['train_micro_batch_size_per_gpu']
        if not isinstance(batch_size, int) or batch_size <= 0:
            logger.warning(f"Invalid batch size: {batch_size}")
            return False
        
        return True


def create_ds_config_file(
    stage: int = 2,
    output_path: str = "ds_config.json",
    **kwargs
) -> Dict[str, Any]:
    """
    Quick helper to create and save a DeepSpeed config.
    
    Args:
        stage: Use ZeRO stage 2 or 3
        output_path: Where to save the config file
        **kwargs: Passed to zero_stage_2() or zero_stage_3()
    
    Returns:
        The config dict (also saved to file)
    """
    if stage == 2:
        config = DeepSpeedConfig.zero_stage_2(**kwargs)
    elif stage == 3:
        config = DeepSpeedConfig.zero_stage_3(**kwargs)
    else:
        raise ValueError(f"Invalid stage: {stage}. Use 2 or 3.")
    
    # Validate before saving
    if not DeepSpeedConfig.validate_config(config):
        logger.warning("Config validation failed but continuing anyway")
    
    DeepSpeedConfig.save_config(config, output_path)
    return config


# Quick reference configs for common scenarios

SMALL_MODEL_CONFIG = DeepSpeedConfig.zero_stage_2(
    micro_batch_size=8,
    gradient_accumulation_steps=4
)

MEDIUM_MODEL_CONFIG = DeepSpeedConfig.zero_stage_2(
    micro_batch_size=4,
    gradient_accumulation_steps=8,
    offload_optimizer=True
)

LARGE_MODEL_CONFIG = DeepSpeedConfig.zero_stage_3(
    micro_batch_size=2,
    gradient_accumulation_steps=16,
    offload_optimizer=True,
    offload_param=True
)

HUGE_MODEL_CONFIG = DeepSpeedConfig.zero_stage_3(
    micro_batch_size=1,
    gradient_accumulation_steps=32,
    offload_optimizer=True,
    offload_param=True,
    sub_group_size=5e8
)
