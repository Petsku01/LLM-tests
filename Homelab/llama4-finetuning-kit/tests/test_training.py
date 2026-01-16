"""Unit tests for training utilities."""

import pytest
from finetune_llama4_company import validate_dataset, estimate_vram_usage


class TestDatasetValidation:
    """Test dataset validation logic."""
    
    def test_missing_file(self, tmp_path):
        """Non-existent file should fail."""
        result = validate_dataset(str(tmp_path / "missing.json"))
        assert result is False
    
    def test_empty_dataset(self, tmp_path):
        """Empty dataset should fail."""
        import json
        file = tmp_path / "empty.json"
        file.write_text(json.dumps([]))
        
        result = validate_dataset(str(file))
        assert result is False
    
    def test_invalid_format(self, tmp_path):
        """Dataset without conversations should fail."""
        import json
        file = tmp_path / "invalid.json"
        file.write_text(json.dumps([{"text": "no conversations"}]))
        
        result = validate_dataset(str(file))
        assert result is False
    
    def test_valid_dataset(self, tmp_path):
        """Valid ShareGPT dataset should pass."""
        import json
        file = tmp_path / "valid.json"
        data = [{
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello"}
            ]
        }]
        file.write_text(json.dumps(data))
        
        result = validate_dataset(str(file))
        assert result is True


class TestVRAMEstimation:
    """Test VRAM estimation."""
    
    def test_4bit_estimation(self):
        """Estimate VRAM for 4-bit model."""
        config = {
            "load_in_4bit": True,
            "max_seq_length": 2048,
            "per_device_train_batch_size": 2,
            "lora_r": 16,
            "use_gradient_checkpointing": "unsloth"
        }
        
        # Should not raise exception
        estimate_vram_usage(config)
    
    def test_16bit_estimation(self):
        """Estimate VRAM for 16-bit model."""
        config = {
            "load_in_4bit": False,
            "max_seq_length": 4096,
            "per_device_train_batch_size": 1,
            "lora_r": 32,
            "use_gradient_checkpointing": "unsloth"
        }
        
        # Should not raise exception
        estimate_vram_usage(config)
