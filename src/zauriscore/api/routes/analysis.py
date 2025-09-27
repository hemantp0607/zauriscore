"""Analysis endpoints for smart contract assessment."""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
import torch
from ...core.config import settings
from ...models.hybrid_model import HybridRiskModel

router = APIRouter()

# Cache the model
_model = None

def get_model():
    """Get or load the ML model."""
    global _model
    if _model is None:
        _model = HybridRiskModel()
        _model.load_state_dict(torch.load(settings.MODEL_PATH, map_location='cpu'))
        _model.eval()
    return _model

class AnalysisRequest:
    def __init__(
        self,
        source_code: str,
        function_count: Optional[int] = None,
        security_flags: Optional[int] = None,
        complexity: Optional[float] = None
    ):
        self.source_code = source_code
        self.function_count = function_count or 0
        self.security_flags = security_flags or 0
        self.complexity = complexity or 0.0

@router.post("/analyze")
async def analyze_contract(
    source_code: str,
    function_count: Optional[int] = None,
    security_flags: Optional[int] = None,
    complexity: Optional[float] = None,
    model = Depends(get_model)
):
    """
    Analyze smart contract code for security risks.
    
    - **source_code**: Smart contract source code
    - **function_count**: Number of functions in the contract (optional)
    - **security_flags**: Number of security flags (optional)
    - **complexity**: Complexity score (0-10) (optional)
    """
    try:
        # Prepare request
        request = AnalysisRequest(
            source_code=source_code,
            function_count=function_count,
            security_flags=security_flags,
            complexity=complexity
        )
        
        # Get prediction (simplified example)
        with torch.no_grad():
            # This is a placeholder - replace with actual model inference
            prediction = 0.5  # Example prediction
            
        return {
            "risk_score": float(prediction),
            "risk_level": "high" if prediction > 0.7 else "medium" if prediction > 0.3 else "low",
            "confidence": min(float(prediction * 1.2), 1.0),
            "features_importance": {
                "code_quality": 0.4,
                "security_issues": 0.3,
                "complexity": 0.3
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
