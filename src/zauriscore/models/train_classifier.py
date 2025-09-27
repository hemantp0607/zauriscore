import os
import numpy as np
import joblib
import subprocess
import json
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - ZauriScore Training',
                    handlers=[logging.FileHandler('model_training.log'), logging.StreamHandler()])

# --- Configuration ---
# Derive project root from this file location: src/zauriscore/models/train_classifier.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
CONTRACTS_DIR = PROJECT_ROOT / "contracts"  # Where .sol files are stored
EMBEDDINGS_FILE = DATA_DIR / "X_embeddings.npy"
LABELS_FILE = DATA_DIR / "y_labels.npy"
FILE_PATHS_FILE = DATA_DIR / "file_paths.npy"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "codebert_vuln_classifier.joblib"

TEST_SIZE = 0.2
RANDOM_STATE = 42

def run_slither(contract_path):
    if not shutil.which('slither'):
        logging.error("Slither not found. Install via 'pip install crytic-slither'.")
        return None
    try:
        result = subprocess.run(
            ["slither", str(contract_path), "--json", "slither_tmp.json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            logging.warning(f"Slither failed on {contract_path}")
            return None
        with open("slither_tmp.json", "r") as f:
            return json.load(f)
    except subprocess.TimeoutExpired:
        logging.warning(f"Slither timeout on {contract_path}")
        return None
    except Exception as e:
        logging.error(f"Error running Slither on {contract_path}: {e}")
        return None

def extract_binary_label(slither_output):
    if not slither_output:
        return 0
    findings = slither_output.get("results", {}).get("detectors", [])
    return 1 if findings else 0

def generate_labels_and_paths():
    logging.info("Generating labels using Slither...")
    labels = []
    paths = []
    contract_files = [f for f in os.listdir(CONTRACTS_DIR) if f.endswith(".sol")]

    for f in contract_files:
        full_path = CONTRACTS_DIR / f
        slither_out = run_slither(full_path)
        label = extract_binary_label(slither_out)
        labels.append(label)
        paths.append(str(full_path))

    y_array = np.array(labels)
    path_array = np.array(paths)
    np.save(LABELS_FILE, y_array)
    np.save(FILE_PATHS_FILE, path_array)
    logging.info(f"Saved {len(labels)} labels and file paths.")

def analyze_feature_importance(classifier, X_train, feature_names=None):
    if hasattr(classifier, 'coef_'):
        # For Logistic Regression, use coefficients
        importances = np.abs(classifier.coef_[0])
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        logging.info("Top 10 Most Important Features:")
        for f in range(min(10, len(indices))):
            logging.info(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")
    else:
        logging.warning("Feature importance not available for this model type")

def analyze_misclassifications(X_test, y_test, y_pred, paths_test, classifier):
    misclassified_indices = np.where(y_test != y_pred)[0]
    
    logging.info("Detailed Misclassification Analysis:")
    for idx in misclassified_indices:
        # Get prediction probabilities
        proba = classifier.predict_proba([X_test[idx]])
        
        logging.info(f"Misclassified Contract: {paths_test[idx]}")
        logging.info(f"True Label: {y_test[idx]}")
        logging.info(f"Predicted Label: {y_pred[idx]}")
        logging.info(f"Prediction Probabilities: {proba}")
        logging.info("---")

def create_advanced_ensemble():
    # Ensemble of multiple classifiers with different strengths
    lr = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=RANDOM_STATE)
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=RANDOM_STATE)
    
    return VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('svm', svm)],
        voting='soft'
    )

def train_classifier(X_train, y_train, X_test=None, y_test=None):
    """
    Train a classifier on the given training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Optional test features for evaluation
        y_test: Optional test labels for evaluation
        
    Returns:
        Trained classifier model and evaluation metrics if test data is provided
    """
    # Create and train the ensemble model
    model = create_advanced_ensemble()
    model.fit(X_train, y_train)
    
    results = {}
    
    # Evaluate on test data if provided
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        logging.info(f"Model trained with accuracy: {results['accuracy']:.4f}")
        logging.info(f"ROC AUC: {results['roc_auc']:.4f}")
    
    return model, results

def main():
    logging.info("Starting advanced classifier training process...")

    # Auto-generate labels if missing
    if not (LABELS_FILE.exists() and FILE_PATHS_FILE.exists()):
        logging.warning("Label or path files missing. Generating with Slither...")
        generate_labels_and_paths()

    # Load data
    if not EMBEDDINGS_FILE.exists():
        logging.error(f"Missing embeddings file: {EMBEDDINGS_FILE}")
        return

    try:
        X = np.load(EMBEDDINGS_FILE)
        y = np.load(LABELS_FILE)
        file_paths = np.load(FILE_PATHS_FILE, allow_pickle=True)
        logging.info(f"Data loaded. X: {X.shape}, y: {y.shape}, paths: {file_paths.shape}")
    except (FileNotFoundError, IOError) as e:
        logging.error(f"Failed to load data: {e}")
        return
    except Exception as e:
        logging.error(f"Unexpected load error: {e}")
        return

    if not (X.shape[0] == y.shape[0] == file_paths.shape[0]):
        logging.error("Mismatch between data shapes.")
        return

    if X.shape[0] == 0:
        logging.error("Empty dataset.")
        return

    # Advanced Sampling Strategy
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),  # Better for ensemble methods
        ('sampler', SMOTETomek(random_state=RANDOM_STATE)),  # Combine over and under sampling
        ('classifier', create_advanced_ensemble())
    ])

    # Stratified K-Fold for more robust evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    # Detailed performance tracking
    cv_scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }

    # Cross-validation with detailed tracking
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        cv_scores['accuracy'].append(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True)
        cv_scores['precision'].append(report['weighted avg']['precision'])
        cv_scores['recall'].append(report['weighted avg']['recall'])
        cv_scores['f1'].append(report['weighted avg']['f1-score'])
        cv_scores['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))

    # Log cross-validation results
    logging.info("Cross-Validation Results:")
    for metric, scores in cv_scores.items():
        logging.info(f"{metric}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    # Final training on full dataset
    pipeline.fit(X, y)

    # Use a train/test split for final eval (instead of last fold)
    X_train_final, X_test_final, y_train_final, y_test_final, paths_train_final, paths_test_final = train_test_split(
        X, y, file_paths, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    pipeline.fit(X_train_final, y_train_final)
    y_pred_final = pipeline.predict(X_test_final)
    y_pred_proba_final = pipeline.predict_proba(X_test_final)[:, 1]
    
    logging.info("Final Classification Report:\n" + classification_report(y_test_final, y_pred_final))
    logging.info("Final Confusion Matrix:\n" + str(confusion_matrix(y_test_final, y_pred_final)))
    
    # Advanced misclassification analysis
    analyze_misclassifications(X_test_final, y_test_final, y_pred_final, paths_test_final, pipeline)

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    logging.info(f"Advanced ensemble model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()