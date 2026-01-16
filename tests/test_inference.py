"""Unit tests for inference utilities."""

import pytest
from unittest.mock import Mock, MagicMock
from inference import format_prompt


class TestPromptFormatting:
    """Test prompt formatting utilities."""
    
    def test_single_message(self):
        """Format single user message."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template = Mock(return_value="formatted")
        
        result = format_prompt(messages, mock_tokenizer)
        assert result == "formatted"
        mock_tokenizer.apply_chat_template.assert_called_once()
    
    def test_multiple_messages(self):
        """Format conversation with multiple turns."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"}
        ]
        
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template = Mock(return_value="conversation")
        
        result = format_prompt(messages, mock_tokenizer)
        assert result == "conversation"
    
    def test_empty_messages(self):
        """Handle empty message list."""
        messages = []
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template = Mock(return_value="")
        
        result = format_prompt(messages, mock_tokenizer)
        assert result == ""
