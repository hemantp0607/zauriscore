"""
ZauriScore Models Package

This package contains all ML model-related functionality:
- CodeBERTRegressor: Custom regression head for vulnerability scoring
- predict_vulnerabilities: Function to predict vulnerabilities in contracts
- train_classifier: Train ensemble classifier
- train_regression_model: Train regression model
- create_advanced_ensemble: Create VotingClassifier ensemble
"""
from .codebert import CodeBERTRegressor
from .predict_vulnerability import predict_vulnerabilities
from .train_classifier import train_classifier, create_advanced_ensemble
from .train_regression import train_regression_model

__all__ = ['CodeBERTRegressor', 'predict_vulnerabilities', 
           'train_classifier', 'train_regression_model', 'create_advanced_ensemble']