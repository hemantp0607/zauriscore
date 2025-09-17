# Model Documentation

## Overview

ZauriScore uses several machine learning models for vulnerability detection:

1. **CodeBERT Regressor**
   - Base model: microsoft/codebert-base
   - Fine-tuned for vulnerability detection
   - Outputs risk score (0-100)

2. **Vulnerability Classifier**
   - Binary classification (vulnerable/safe)
   - Based on CodeBERT architecture
   - High precision on known vulnerabilities

## Model Architecture

### CodeBERT Regressor

```python
class CodeBERTRegressor(torch.nn.Module):
    """
    Neural network for predicting vulnerability risk scores.
    Uses CodeBERT base with regression head.
    """
    def __init__(self, base_model='microsoft/codebert-base'):
        self.codebert = RobertaModel.from_pretrained(base_model)
        self.regressor = nn.Linear(768, 1)
```

### Risk Score Mapping

```python
VULNERABILITY_SCORE_MAPPING = {
    'critical': range(0, 20),    # Critical Risk
    'high': range(20, 40),      # High Risk
    'medium': range(40, 60),    # Medium Risk
    'low': range(60, 80),       # Low Risk
    'safe': range(80, 100)      # Very Safe
}
```

## Training

### Data Format

```json
{
    "contracts": [
        {
            "source": "contract code...",
            "score": 85,
            "vulnerabilities": []
        }
    ]
}
```

### Training Process

1. **Data Preprocessing**
   - Tokenization
   - Padding
   - Augmentation

2. **Training Loop**
   - Batch size: 8
   - Learning rate: 2e-5
   - Epochs: 10

3. **Evaluation**
   - MSE for regression
   - F1-score for classification

## Model Usage

```python
from zauriscore.models import CodeBERTRegressor

# Load model
model = CodeBERTRegressor.load('path/to/model.pth')

# Predict
score = model.predict('contract source code')
```

## Performance Metrics

- MSE: 0.15
- R²: 0.85
- Precision: 0.92
- Recall: 0.88
