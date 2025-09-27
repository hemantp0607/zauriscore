# Usage Guide

## Command Line Interface

ZauriScore provides a powerful CLI for contract analysis and model training.

### Basic Commands

1. **Analyze a Contract**
```bash
# Single contract analysis
zauriscore analyze path/to/contract.sol

# Analyze with detailed report
zauriscore analyze path/to/contract.sol --report

# Analyze multiple contracts
zauriscore analyze path/to/contracts/ --recursive
```

2. **Train Models**
```bash
# Train with default settings
zauriscore train path/to/dataset.json

# Custom training
zauriscore train \
    --dataset path/to/dataset.json \
    --model-output models/new_model.pth \
    --epochs 10 \
    --batch-size 32
```

3. **Web Interface**
```bash
# Start web server
zauriscore serve

# Custom port
zauriscore serve --port 8080
```

## Web Interface

The web interface provides a user-friendly way to analyze contracts:

1. Open http://localhost:8000 in your browser
2. Upload a contract file or paste the source code
3. Click "Analyze"
4. View the detailed report

## Configuration

### Custom Settings

Modify `config/default.yml` to customize:

```yaml
# API Settings
api:
  host: "0.0.0.0"
  port: 8000

# Model Settings
model:
  base_model: "microsoft/codebert-base"
  max_length: 512

# Analysis Settings
analysis:
  risk_thresholds:
    critical: 20
    high: 40
    medium: 60
    low: 80
```

### Environment Variables

Set these in `.env`:

```env
ETHERSCAN_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # Optional
LOG_LEVEL=INFO
```

## Best Practices

1. **Contract Analysis**
   - Always verify source code matches deployed bytecode
   - Use multiple analysis tools for comprehensive checks
   - Review automated findings manually

2. **Model Training**
   - Use diverse training data
   - Validate model performance on test set
   - Monitor for overfitting

3. **Security**
   - Keep API keys secure
   - Regularly update dependencies
   - Monitor system resources
