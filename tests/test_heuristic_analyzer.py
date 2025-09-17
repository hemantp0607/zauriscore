"""
Tests for the heuristic_analyzer module.
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the module to test
from zauriscore.analyzers.heuristic_analyzer import (
    MLVulnerabilityWeightCalculator,
    normalize_score,
    HEURISTIC_POINTS
)

# Test data
SAMPLE_CONTRACT = """
pragma solidity ^0.8.0;

contract SampleContract {
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    function transferOwnership(address newOwner) public {
        require(msg.sender == owner, "Only owner can transfer ownership");
        owner = newOwner;
    }
}
"""

class TestMLVulnerabilityWeightCalculator:
    """Test cases for MLVulnerabilityWeightCalculator class."""
    
    @pytest.fixture
    def mock_ml_dependencies(self):
        """Mock ML dependencies for testing."""
        with patch('zauriscore.analyzers.heuristic_analyzer.ML_DEPENDENCIES_AVAILABLE', True):
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                with patch('transformers.AutoModel.from_pretrained') as mock_model:
                    # Setup mock model and tokenizer
                    mock_tokenizer.return_value = MagicMock()
                    mock_model.return_value = MagicMock()
                    
                    # Mock the model's forward pass
                    mock_output = MagicMock()
                    mock_output.last_hidden_state = MagicMock()
                    mock_output.last_hidden_state.mean.return_value = MagicMock()
                    mock_output.last_hidden_state.mean.return_value.flatten.return_value = [0.1, 0.2, 0.3]
                    mock_model.return_value.return_value = mock_output
                    
                    yield

    def test_init_with_ml_deps_available(self, mock_ml_dependencies):
        """Test initialization when ML dependencies are available."""
        try:
            calculator = MLVulnerabilityWeightCalculator()
            assert calculator is not None
        except ImportError:
            pytest.fail("MLVulnerabilityWeightCalculator raised ImportError unexpectedly!")
    
    def test_embed_vulnerability_type(self, mock_ml_dependencies):
        """Test _embed_vulnerability_type method."""
        calculator = MLVulnerabilityWeightCalculator()
        embedding = calculator._embed_vulnerability_type("test description")
        assert len(embedding) > 0
    
    def test_calculate_code_vulnerability_similarity(self, mock_ml_dependencies):
        """Test calculate_code_vulnerability_similarity method."""
        calculator = MLVulnerabilityWeightCalculator()
        similarities = calculator.calculate_code_vulnerability_similarity(SAMPLE_CONTRACT)
        
        assert isinstance(similarities, dict)
        assert 'reentrancy' in similarities
        assert 'access_control' in similarities
        assert 'selfdestruct' in similarities
        assert 'external_call' in similarities
        assert 'economic_manipulation' in similarities
    
    def test_assess_economic_risk(self, mock_ml_dependencies):
        """Test assess_economic_risk method."""
        calculator = MLVulnerabilityWeightCalculator()
        risk_level = calculator.assess_economic_risk(SAMPLE_CONTRACT)
        
        assert risk_level in ['economic_risk_low', 'economic_risk_medium', 'economic_risk_high']


class TestHelperFunctions:
    """Test cases for helper functions in heuristic_analyzer."""
    
    def test_normalize_score(self):
        """Test the normalize_score function."""
        # Test normal case
        assert 75.0 == normalize_score(25, 0, 100)
        
        # Test minimum score
        assert 0.0 == normalize_score(-100, -100, 100)
        
        # Test maximum score
        assert 100.0 == normalize_score(100, -100, 100)
        
        # Test clamp to 0
        assert 0.0 == normalize_score(-200, -100, 100)
        
        # Test clamp to 100
        assert 100.0 == normalize_score(200, -100, 100)
        
        # Test invalid range (should return 50.0 as default)
        assert 50.0 == normalize_score(50, 100, 0)


if __name__ == "__main__":
    pytest.main([__file__])
