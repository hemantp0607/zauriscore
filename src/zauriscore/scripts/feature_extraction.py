import pandas as pd
from transformers import CodeBERTModel, AutoTokenizer
import torch

class FeatureExtractor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        self.model = CodeBERTModel.from_pretrained('microsoft/codebert-base')
    
    def extract(self, contract_code):
        metrics = {
            'function_count': len(contract_code.split('function')),
            'security_flags': 1 if 'revert' in contract_code.lower() else 0,
            'complexity': len(contract_code.split('if'))
        }
        
        inputs = self.tokenizer(contract_code, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy().flatten()
        
        return pd.DataFrame([{**metrics, 'embeddings': embeddings}])

if __name__ == '__main__':
    extractor = FeatureExtractor()
    result = extractor.extract(open('sample.sol').read())
    result.to_parquet('output/contract_features.parquet')