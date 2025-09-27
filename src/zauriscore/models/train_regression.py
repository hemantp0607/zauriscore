import os
import torch
import numpy as np
import logging
from transformers import RobertaTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.nn import MSELoss
from sklearn.model_selection import train_test_split
from pathlib import Path

from .codebert import CodeBERTRegressor
from ..data.data_processor import DataProcessor

class ContractDatasetBuilder:
    def __init__(self):
        try:
            self.processor = DataProcessor()
        except ImportError:
            logging.warning("DataProcessor unavailable; using mock fetch.")
            self.processor = None

    def fetch_contract_code(self, address):
        if self.processor:
            # Assume DataProcessor has fetch_contract_code; implement if needed
            return self.processor.fetch_contract_code(address)  # Stub: return mock
        # Mock fallback
        logging.info(f"Mock code for {address}")
        return f"contract Mock{address[-4:]} {{ }}"

    def collect_contracts(self, safe_addresses, vulnerable_addresses):
        all_contracts = []
        all_scores = []  # For regression: e.g., 0.1 for safe, 0.9 for vuln
        for addr in safe_addresses:
            code = self.fetch_contract_code(addr)
            if code:
                all_contracts.append(code)
                all_scores.append(0.1)  # Low score
        for addr in vulnerable_addresses:
            code = self.fetch_contract_code(addr)
            if code:
                all_contracts.append(code)
                all_scores.append(0.9)  # High score
        logging.info(f"Collected {len(all_contracts)} contracts.")
        return all_contracts, all_scores

class ContractRegressionDataset(Dataset):
    def __init__(self, contracts, scores, tokenizer, max_length=512):
        self.contracts = contracts
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contracts)

    def __getitem__(self, idx):
        contract = self.contracts[idx]
        try:
            encoding = self.tokenizer(
                contract, 
                truncation=True, 
                padding='max_length', 
                max_length=self.max_length, 
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'score': self.scores[idx]
            }
        except Exception as e:
            logging.error(f"Tokenization failed for contract {idx}: {e}")
            # Return dummy to avoid crash
            return {
                'input_ids': torch.zeros(self.max_length),
                'attention_mask': torch.zeros(self.max_length),
                'score': self.scores[idx]
            }

def train_regression_model(epochs=10, batch_size=8, learning_rate=2e-5):
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - ZauriScore™ Regression Training',
        filename='regression_training.log',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Load dataset
    dataset_builder = ContractDatasetBuilder()
    contracts, scores = dataset_builder.collect_contracts(
        safe_addresses=[
            '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',  # Uniswap Router
            '0xdAC17F958D2ee523a2206206994597C13D831ec7',  # Tether
        ],
        vulnerable_addresses=[
            '0x82E47839E24d669a760fC5E59Af44F4049C66b4C',
            '0xf56a4A9ae94aeE8F6e9BF737907389c94285AbE6'
        ]
    )

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        contracts, scores, test_size=0.2, random_state=42
    )

    # Initialize model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = CodeBERTRegressor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create datasets
    train_dataset = ContractRegressionDataset(X_train, y_train, tokenizer)
    test_dataset = ContractRegressionDataset(X_test, y_test, tokenizer)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Optimizer and Loss
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = MSELoss()

    # Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)

            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, scores)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}')

    # Evaluation
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            scores = batch['score'].to(device)

            predictions = model(input_ids, attention_mask)
            test_loss = criterion(predictions, scores)
            total_test_loss += test_loss.item()
    
    print(f'Test Loss: {total_test_loss/len(test_loader)}')

    # Save model
    models_dir = Path(__file__).parent.parent / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), models_dir / 'zauriscore_regression_model.pth')
    print("Model saved successfully!")

if __name__ == '__main__':
    train_regression_model()