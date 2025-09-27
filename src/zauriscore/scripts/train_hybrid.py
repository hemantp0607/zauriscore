import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

# Load dataset
DATA_PATH = Path('output/contract_features.parquet')
df = pd.read_parquet(DATA_PATH)

# Prepare data
X_numeric = df[['function_count', 'security_flags', 'complexity']].values
X_bert = df['embeddings'].values
y = df['risk_score'].values

# Split data
train_x_num, val_x_num, train_x_bert, val_x_bert, train_y, val_y = train_test_split(X_numeric, X_bert, y, test_size=0.2)

# Initialize model
model = HybridRiskModel()

# Training setup
trainer = pl.Trainer(max_epochs=50, early_stopping=True)

# Training loop
trainer.fit(model, train_dataloaders=pl.utils.setup_dataloaders(train_x_num, train_x_bert, train_y))

# Save best model
torch.save(model.state_dict(), 'best_model.pth')