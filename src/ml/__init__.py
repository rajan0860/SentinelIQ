"""
ML Package
==========
Ensemble fraud detection models and feature engineering.

EnsembleScorer        → XGBoost (70%) + Isolation Forest (30%) risk score
FraudScorer           → EnsembleScorer + SHAP explanations
GraphFeatureExtractor → degree centrality, component size, shared device/IP counts
build_feature_matrix  → combines tabular + graph features for training
train()               → full training pipeline with SMOTE
"""
