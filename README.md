# Overview
LDA-SCGB: Predicting lncRNA-disease associations with graph autoencoder and noise-robust gradient boosting
# Dataset
In this work，lncRNADisease is Dataset 1 and MNDR is Dataset 2.

# Environment
Install python3.7 for running this model. And these packages should be satisfied:
- numpy $\approx$ 1.23.2
- padas $\approx$ 2.1.4
- scikit-learn $\approx$ 1.3.0
- xgboost=2.0.0
# Extracting linear features for diseases and lncRNAs by NMF, to run:：
- python linear_feature.py
# Extracting nonlinear features for diseases and lncRNAs by GAE, to run:：
- python ./nonlinear_feature/main.py
# To run the model fastly：
- python mian.py
