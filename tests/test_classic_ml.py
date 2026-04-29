import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from src.preprocessing import TumorRowlandPreprocessor
from src.vectorizer import TumorRowlandVectorizer
from models.classic_ml import TumorRowlandClassicModels

def run_experiment():
    if torch.cuda.is_available():
        print(f"\n[SISTEMA] GPU detectada y disponible: {torch.cuda.get_device_name(0)}\n")

    df = pd.read_csv('data/tcga_simple_train.csv')
    
    preprocessor = TumorRowlandPreprocessor()
    vectorizer = TumorRowlandVectorizer(max_features=5000, max_len=500)
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['t'])
    
    X_train_clean = df_train['text'].apply(preprocessor.clean_for_sequence_models)
    X_test_clean = df_test['text'].apply(preprocessor.clean_for_sequence_models)
    
    X_train_vec, X_test_vec = vectorizer.vectorize_for_trees(X_train_clean, X_test_clean)
    
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(df_train['t'])
    y_test = encoder.transform(df_test['t'])
    
    models_manager = TumorRowlandClassicModels()
    
    print("Iniciando fase de entrenamiento...")
    models_manager.train_all(X_train_vec, y_train)
    
    print("Generando probabilidades para Soft Voting...")
    probas = models_manager.predict_proba_all(X_test_vec)
    
    weights = {
        'logistic': 1.0,
        'random_forest': 1.0,
        'naive_bayes': 1.0,
        'xgboost': 2.0
    }
    
    weighted_probas = (
        probas['logistic'] * weights['logistic'] +
        probas['random_forest'] * weights['random_forest'] +
        probas['naive_bayes'] * weights['naive_bayes'] +
        probas['xgboost'] * weights['xgboost']
    )
    
    soft_voting_preds = np.argmax(weighted_probas, axis=1)
    
    print("\n" + "="*45)
    print(" COMPARATIVA: HARD VS SOFT VOTING PONDERADO ")
    print("="*45)
    
    xgb_score = f1_score(y_test, np.argmax(probas['xgboost'], axis=1), average='macro')
    print(f"| {'solo_xgboost'.ljust(20)} | Macro-F1: {xgb_score:.4f} |")
    
    soft_score = f1_score(y_test, soft_voting_preds, average='macro')
    print(f"| {'soft_voting_weighted'.ljust(20)} | Macro-F1: {soft_score:.4f} |")
    print("="*45)
    
    print(f"\nReporte detallado para SOFT VOTING PONDERADO:")
    print(classification_report(y_test, soft_voting_preds, target_names=encoder.classes_))

if __name__ == "__main__":
    run_experiment()