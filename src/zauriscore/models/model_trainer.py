import os
import torch
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

# ML imports
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel

# Local imports with graceful fallback
try:
    from ..data import DataProcessor
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("DataProcessor unavailable; using placeholder.")
    class DataProcessor:
        @staticmethod
        def process(*args, **kwargs):
            return [], []

# Configure logging
logger = logging.getLogger(__name__)

# Project root is two levels up from this file
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
DATA_DIR = PROJECT_ROOT / 'datasets'
MODELS_DIR = PROJECT_ROOT / 'models'
LOGS_DIR = PROJECT_ROOT / 'logs'

# Create directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure file logging
log_file = LOGS_DIR / 'model_training.log'
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

class CodeBERTRegressorHead(torch.nn.Module):
    VULNERABILITY_SCORE_MAPPING = {
        'critical': range(0, 20),     # Critical Risk
        'high': range(20, 40),        # High Risk
        'medium': range(40, 60),      # Medium Risk
        'low': range(60, 80),         # Low Risk
        'safe': range(80, 100)        # Very Safe
    }

    def __init__(self, base_model='microsoft/codebert-base'):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.codebert = RobertaModel.from_pretrained(base_model).to(self.device)
        
        # Freeze CodeBERT base parameters
        for param in self.codebert.parameters():
            param.requires_grad = False
        
        # Advanced regression head with more layers and dropout
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(self.codebert.config.hidden_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 1)
        ).to(self.device)
    
    def forward(self, input_ids, attention_mask):
        # Ensure input is 2D tensor
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        
        outputs = self.codebert(
            input_ids=input_ids.to(self.device), 
            attention_mask=attention_mask.to(self.device)
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Handle single sample case
        if pooled_output.size(0) == 1:
            # Add a dummy batch dimension for batch normalization
            pooled_output = torch.cat([pooled_output, pooled_output], dim=0)
            regression_output = self.regressor(pooled_output)
            return regression_output[0]
        
        return self.regressor(pooled_output).squeeze()

    @classmethod
    def generate_vulnerability_score(cls, risk_category='medium'):
        if risk_category not in cls.VULNERABILITY_SCORE_MAPPING:
            risk_category = 'medium'
        score_range = list(cls.VULNERABILITY_SCORE_MAPPING[risk_category])
        return np.random.choice(score_range)

    @classmethod
    def categorize_risk_score(cls, score):
        for category, score_range in cls.VULNERABILITY_SCORE_MAPPING.items():
            if score in score_range:
                return category.capitalize()
        return 'Unknown Risk'

class SmartContractModelTrainer:
    VULNERABILITY_SCORE_MAPPING = {
        'critical': range(0, 20),     # Critical Risk
        'high': range(20, 40),        # High Risk
        'medium': range(40, 60),      # Medium Risk
        'low': range(60, 80),         # Low Risk
        'safe': range(80, 100)        # Very Safe
    }
    
    @classmethod
    def generate_vulnerability_score(cls, risk_category='medium'):
        """Generate a nuanced vulnerability score based on risk category"""
        if risk_category not in cls.VULNERABILITY_SCORE_MAPPING:
            logging.warning(f'Unknown risk category: {risk_category}. Defaulting to medium risk.')
            risk_category = 'medium'
        
        score_range = cls.VULNERABILITY_SCORE_MAPPING[risk_category]
        return np.random.choice(list(score_range))
    
    @staticmethod
    def analyze_contract_complexity(contract_code):
        """Analyze contract complexity as a feature for risk assessment"""
        metrics = {
            'line_count': len(contract_code.splitlines()),
            'function_count': contract_code.count('function '),
            'modifier_count': contract_code.count('modifier '),
            'external_call_count': contract_code.count('.call('),
            'has_selfdestruct': 'selfdestruct' in contract_code,
            'has_delegatecall': 'delegatecall' in contract_code
        }
        return metrics
    
    def __init__(self, base_model='microsoft/codebert-base'):
        """Initialize the trainer with advanced configurations"""
        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)
        self.base_model = RobertaModel.from_pretrained(base_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cv_strategy = None  # For PyTorch-only; set to StratifiedKFold if sklearn CV needed
        logging.info(f'Using device: {self.device}')
        
        # Initialize classification model
        self.model = RobertaForSequenceClassification.from_pretrained(
            base_model, 
            num_labels=2,  # Binary classification
        ).to(self.device)
    
    def advanced_model_evaluation(self, test_dataset):
        """Perform comprehensive model evaluation - PyTorch implementation"""
        self.model.eval()
        
        with torch.no_grad():
            input_ids = test_dataset['input_ids'].to(self.device)
            attention_mask = test_dataset['attention_mask'].to(self.device)
            true_labels = test_dataset['labels'].cpu().numpy()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            pred_proba = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        roc_auc = roc_auc_score(true_labels, pred_proba)
        cm = confusion_matrix(true_labels, predictions)
        
        performance_metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(true_labels, predictions, output_dict=True),
            'note': 'PyTorch-based evaluation (CV not directly compatible; consider manual folds)'
        }
        
        # Log performance details
        logging.info('Advanced Model Evaluation Results:')
        logging.info(json.dumps(performance_metrics, indent=2))
        
        return performance_metrics
    
    def load_dataset(self, dataset_path):
        """
        Load smart contract dataset
        Expected JSON format: 
        [
            {"code": "contract...", "label": 0/1},
            ...
        ]
        """
        try:
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            codes = [entry['code'] for entry in dataset]
            labels = [entry['label'] for entry in dataset]
            
            if not codes or not labels:
                raise ValueError("Loaded dataset is empty")
            
            return codes, labels
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logging.error(f"Dataset loading error: {e}")
            raise
    
    def prepare_dataset(self, codes, labels, max_length=512, regression=False):
        """
        Tokenize and prepare dataset for training
        """
        encodings = self.tokenizer(
            codes, 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.float32 if regression else torch.long)
        }
    
    def train(self, train_dataset, val_dataset=None, epochs=3, batch_size=8):
        """
        Train the model with optional validation
        """
        # Prepare optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for i in range(0, len(train_dataset['input_ids']), batch_size):
                batch_input_ids = train_dataset['input_ids'][i:i+batch_size].to(self.device)
                batch_attention_mask = train_dataset['attention_mask'][i:i+batch_size].to(self.device)
                batch_labels = train_dataset['labels'][i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / (len(train_dataset['input_ids']) // batch_size)
            logging.info(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        return self.model
    
    def evaluate(self, test_dataset):
        """
        Evaluate model performance
        """
        self.model.eval()
        
        with torch.no_grad():
            input_ids = test_dataset['input_ids'].to(self.device)
            attention_mask = test_dataset['attention_mask'].to(self.device)
            true_labels = test_dataset['labels'].cpu().numpy()
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        
        return classification_report(true_labels, predictions)
    
    def save_model(self, save_path):
        """
        Save trained model and tokenizer
        """
        try:
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logging.info(f'Model saved to {save_path}')
        except Exception as e:
            logging.error(f"Model save failed: {e}")

def display_menu():
    print("ZauriScore Model Training")
    print("1. Classification Model")
    print("2. Regression Model")
    print("3. Exit")

def train_regression_model():
    try:
        # Use relative paths
        dataset_path = DATA_DIR / 'smart_contract_dataset.json'
        model_save_path = MODELS_DIR / 'codebert_regressor'
        
        if not dataset_path.exists():
            logging.error(f"Dataset not found at {dataset_path}")
            return False
        
        # Initialize trainer
        trainer = SmartContractModelTrainer()
        
        # Load dataset
        codes, labels = trainer.load_dataset(dataset_path)
        
        # Generate vulnerability scores
        vulnerability_scores = [trainer.generate_vulnerability_score(label) for label in labels]
        
        # Split dataset
        train_codes, test_codes, train_scores, test_scores = train_test_split(
            codes, vulnerability_scores, test_size=0.2, random_state=42
        )
        
        # Initialize regression model
        regression_model = CodeBERTRegressorHead()
        
        # Prepare datasets
        train_dataset = trainer.prepare_dataset(train_codes, train_scores, regression=True)
        test_dataset = trainer.prepare_dataset(test_codes, test_scores, regression=True)
        
        # Training configuration
        optimizer = torch.optim.AdamW(regression_model.parameters(), lr=2e-5)
        loss_fn = torch.nn.MSELoss()
        
        # Training loop
        regression_model.train()
        for epoch in range(10):
            total_loss = 0
            for i in range(0, len(train_dataset['input_ids']), 8):
                batch_input_ids = train_dataset['input_ids'][i:i+8]
                batch_attention_mask = train_dataset['attention_mask'][i:i+8]
                batch_labels = train_dataset['labels'][i:i+8]
                
                optimizer.zero_grad()
                
                batch_input_ids = batch_input_ids.to(regression_model.device)
                batch_attention_mask = batch_attention_mask.to(regression_model.device)
                batch_labels = batch_labels.to(regression_model.device).unsqueeze(1)
                
                predictions = regression_model(batch_input_ids, batch_attention_mask)
                loss = loss_fn(predictions, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_dataset["input_ids"])}')
        
        # Save model
        torch.save(regression_model.state_dict(), model_save_path)
        print(f'Regression model saved to {model_save_path}')
        
        return True
        
    except Exception as e:
        logging.error(f"Regression model training failed: {e}")
        return False

def train_classification_model():
    try:
        # Use relative paths
        dataset_path = DATA_DIR / 'smart_contract_dataset.json'
        model_save_path = MODELS_DIR / 'codebert_classifier'
        
        if not dataset_path.exists():
            logging.error(f"Dataset not found at {dataset_path}")
            return False
        
        # Initialize trainer
        trainer = SmartContractModelTrainer()
        
        # Load dataset
        codes, labels = trainer.load_dataset(dataset_path)
        
        # Split dataset
        train_codes, test_codes, train_labels, test_labels = train_test_split(
            codes, labels, test_size=0.2, random_state=42
        )
        
        # Prepare datasets
        train_dataset = trainer.prepare_dataset(train_codes, train_labels)
        test_dataset = trainer.prepare_dataset(test_codes, test_labels)
        
        # Train model
        trained_model = trainer.train(train_dataset)
        
        # Evaluate model
        performance_report = trainer.evaluate(test_dataset)
        logging.info('Model Performance Report:\n' + performance_report)
        
        # Save model
        trainer.save_model(model_save_path)
        
        return True
        
    except Exception as e:
        logging.error(f"Classification model training failed: {e}")
        return False

def main():
    display_menu()
    while True:
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == '1':
            try:
                train_classification_model()
            except Exception as e:
                logger.error(f"Classification failed: {e}")
        elif choice == '2':
            try:
                train_regression_model()
            except Exception as e:
                logger.error(f"Regression failed: {e}")
        elif choice == '3':
            print("Exiting ZauriScore Model Training...")
            break
        else:
            print("Invalid choice. Please try again.")
            display_menu()

if __name__ == '__main__':
    main()