import torch
import pytorch_lightning as pl
from transformers import CodeBERTModel, AutoTokenizer

class HybridRiskModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Structured features branch
        self.numeric_layer = nn.Linear(15, 64)
        # CodeBERT embeddings branch
        self.bert = CodeBERTModel.from_pretrained('microsoft/codebert-base')
        self.embedding_layer = nn.Linear(768, 64)
        # Final layer
        self.fc = nn.Linear(128, 1)
    
    def forward(self, numeric_features, code_tokens):
        # Process numeric features
        x = self.numeric_layer(numeric_features)
        
        # Process code tokens
        with torch.no_grad():
            bert_outputs = self.bert(code_tokens)
        embeddings = bert_outputs.last_hidden_state.mean(dim=1)
        x = torch.cat([x, self.embedding_layer(embeddings)], dim=1)
        
        return torch.sigmoid(self.fc(x))