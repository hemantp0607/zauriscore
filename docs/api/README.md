# ZauriScore API Documentation

## Overview

ZauriScore provides both a Python API and a REST API for smart contract security analysis.

## Python API

### Analyzers

```python
from zauriscore.analyzers import ComprehensiveContractAnalyzer

# Initialize analyzer
analyzer = ComprehensiveContractAnalyzer()

# Analyze a contract
result = analyzer.analyze_contract('path/to/contract.sol')
```

### ML Pipeline

```python
from zauriscore import ZauriScoreMLPipeline

# Initialize pipeline
pipeline = ZauriScoreMLPipeline()

# Train model
pipeline.train('path/to/dataset.json', 'path/to/save/model.pth')

# Predict vulnerabilities
score = pipeline.predict('path/to/contract.sol')
```

## REST API

### Endpoints

#### POST /analyze
Analyze a smart contract for vulnerabilities.

**Request:**
```json
{
    "contract_source": "string",
    "address": "string" (optional)
}
```

**Response:**
```json
{
    "score": float,
    "vulnerabilities": [
        {
            "type": "string",
            "severity": "string",
            "description": "string",
            "line_number": int
        }
    ],
    "recommendations": [
        {
            "title": "string",
            "description": "string"
        }
    ]
}
```

#### GET /status
Get the service status.

**Response:**
```json
{
    "status": "string",
    "version": "string"
}
```
