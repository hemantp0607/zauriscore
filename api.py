from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer

app = FastAPI()

# Load saved model
model = HybridRiskModel()
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')

# Request models
class PredictionRequest(BaseModel):
    source_code: str
    structured_features: dict = None

# Explanation models
import shap

@app.post('/predict')
def predict(request: PredictionRequest):
    # Process input
    if request.structured_features:
        numeric_features = np.array([
            request.structured_features.get('function_count', 0),
            request.structured_features.get('security_flags', 0),
            request.structured_features.get('complexity', 0)
        ])
    else:
        numeric_features = np.zeros(3)
    
    # Tokenize code
    inputs = tokenizer(request.source_code, return_tensors='pt', padding=True, truncation=True)
    code_tokens = inputs['input_ids']
    
    # Get predictions
    with torch.no_grad():
        numeric_input = torch.tensor(numeric_features, dtype=torch.float32)
        code_input = torch.tensor(code_tokens, dtype=torch.long)
        prediction = model(numeric_input, code_input)
    
    # SHAP explainability
    explainer = shap.DeepExplainer(model, numeric_features)
    shap_values = explainer.shap_values(numeric_features)
    
    return {
        'risk_score': prediction.item(),
        'shap_values': shap_values.tolist()
    }