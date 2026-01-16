"""Unit tests for dataset utilities."""

import json
import pytest
from pathlib import Path
from scripts.prepare_dataset import (
    validate_sharegpt_format,
    convert_alpaca_to_sharegpt,
    convert_oasst_to_sharegpt,
)


class TestShareGPTValidation:
    """Test ShareGPT format validation."""
    
    def test_valid_format(self):
        """Valid ShareGPT sample should pass."""
        sample = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi there!"}
            ]
        }
        assert validate_sharegpt_format(sample) is True
    
    def test_missing_conversations(self):
        """Sample without conversations key should fail."""
        sample = {"messages": []}
        assert validate_sharegpt_format(sample) is False
    
    def test_empty_conversations(self):
        """Empty conversations list should fail."""
        sample = {"conversations": []}
        assert validate_sharegpt_format(sample) is False
    
    def test_invalid_conversation_structure(self):
        """Conversation without required fields should fail."""
        sample = {
            "conversations": [
                {"role": "user", "content": "test"}
            ]
        }
        assert validate_sharegpt_format(sample) is False
    
    def test_invalid_role(self):
        """Invalid role should fail."""
        sample = {
            "conversations": [
                {"from": "invalid_role", "value": "test"}
            ]
        }
        assert validate_sharegpt_format(sample) is False


class TestAlpacaConversion:
    """Test Alpaca to ShareGPT conversion."""
    
    def test_basic_conversion(self):
        """Convert basic Alpaca sample."""
        alpaca = {
            "instruction": "What is AI?",
            "input": "",
            "output": "AI is artificial intelligence."
        }
        result = convert_alpaca_to_sharegpt(alpaca)
        
        assert "conversations" in result
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"
    
    def test_conversion_with_input(self):
        """Convert Alpaca sample with input field."""
        alpaca = {
            "instruction": "Summarize this",
            "input": "Long text here",
            "output": "Summary"
        }
        result = convert_alpaca_to_sharegpt(alpaca)
        
        user_message = result["conversations"][0]["value"]
        assert "Summarize this" in user_message
        assert "Long text here" in user_message


class TestOASSTConversion:
    """Test OASST to ShareGPT conversion."""
    
    def test_basic_conversion(self):
        """Convert OASST format."""
        oasst = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"}
            ]
        }
        result = convert_oasst_to_sharegpt(oasst)
        
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["from"] == "human"
        assert result["conversations"][1]["from"] == "gpt"
    
    def test_system_message(self):
        """Handle system messages."""
        oasst = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"}
            ]
        }
        result = convert_oasst_to_sharegpt(oasst)
        
        assert result["conversations"][0]["from"] == "system"
