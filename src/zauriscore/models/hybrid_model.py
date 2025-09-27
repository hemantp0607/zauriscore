import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import CodeBERTModel, AutoTokenizer

class HybridRiskModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Structured features branch (assumes 15 numeric features)
        self.numeric_layer = nn.Linear(15, 64)
        # CodeBERT embeddings branch (768 hidden dim)
        self.bert = CodeBERTModel.from_pretrained('microsoft/codebert-base')
        self.bert.requires_grad_(False)  # Freeze BERT
        self.embedding_layer = nn.Linear(768, 64)
        # Final layer
        self.fc = nn.Linear(128, 1)
    
    def forward(self, numeric_features, code_tokens):
        # Device handling
        device = next(self.parameters()).device
        numeric_features = numeric_features.to(device)
        code_tokens = {k: v.to(device) for k, v in code_tokens.items()}
        
        # Process numeric features
        x = F.relu(self.numeric_layer(numeric_features))
        
        # Process code tokens
        with torch.no_grad():
            bert_outputs = self.bert(**code_tokens)
        embeddings = bert_outputs.last_hidden_state.mean(dim=1)
        x = torch.cat([x, self.embedding_layer(embeddings)], dim=1)
        
        return torch.sigmoid(self.fc(x))

    def training_step(self, batch, batch_idx):
        numeric, code_tokens, targets = batch  # Assume dataloader yields (numeric, tokens_dict, targets)
        preds = self(numeric, code_tokens)
        loss = F.binary_cross_entropy(preds, targets.float().unsqueeze(1))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        numeric, code_tokens, targets = batch
        preds = self(numeric, code_tokens)
        loss = F.binary_cross_entropy(preds, targets.float().unsqueeze(1))
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)