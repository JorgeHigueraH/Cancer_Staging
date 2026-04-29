import pandas as pd
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import copy

from src.preprocessing import TumorRowlandPreprocessor
from src.vectorizer import TumorRowlandVectorizer
from models.classic_ml import TumorRowlandClassicModels
from models.sequence_models import TumorRowlandCNN, TumorRowlandLSTM

def train_dl_model(model, train_loader, val_loader, criterion, optimizer, device):
    best_loss = float('inf')
    best_wts = None
    
    for epoch in range(15):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                v_loss += criterion(model(x.to(device)), y.to(device)).item()
        v_loss /= len(val_loader)
        
        if v_loss < best_loss:
            best_loss = v_loss
            best_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_wts)
    return model

def run_full_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SISTEMA] Dispositivo: {device}")

    df = pd.read_csv('data/tcga_simple_train.csv')
    pre = TumorRowlandPreprocessor()
    vec = TumorRowlandVectorizer(max_features=5000, max_len=500)
    
    df_tr, df_ts = train_test_split(df, test_size=0.2, random_state=42, stratify=df['t'])
    
    X_tr_c = df_tr['text'].apply(pre.clean_for_sequence_models)
    X_ts_c = df_ts['text'].apply(pre.clean_for_sequence_models)
    
    X_tr_tree, X_ts_tree = vec.vectorize_for_trees(X_tr_c, X_ts_c)
    X_tr_seq, X_ts_seq = vec.vectorize_for_sequence_models(X_tr_c, X_ts_c)
    
    enc = LabelEncoder()
    y_tr = enc.fit_transform(df_tr['t'])
    y_ts = enc.transform(df_ts['t'])

    cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr)
    cw_t = torch.tensor(cw, dtype=torch.float32).to(device)

    try:
        wv = KeyedVectors.load_word2vec_format('data/bio_embedding_intrinsic.bin', binary=True)
        mtrx = vec.get_bioword2vec_matrix(wv)
    except:
        mtrx = np.zeros((5000, 200))

    cl_manager = TumorRowlandClassicModels()
    cl_manager.train_all(X_tr_tree, y_tr)
    cl_probas = cl_manager.predict_proba_all(X_ts_tree)

    X_t = torch.tensor(X_tr_seq, dtype=torch.long)
    y_t = torch.tensor(y_tr, dtype=torch.long)
    split = int(0.9 * len(X_t))
    tr_loader = DataLoader(TensorDataset(X_t[:split], y_t[:split]), batch_size=64, shuffle=True)
    vl_loader = DataLoader(TensorDataset(X_t[split:], y_t[split:]), batch_size=64)

    dl_models = {
        'cnn': TumorRowlandCNN(5000, 200, 4, mtrx).to(device),
        'lstm': TumorRowlandLSTM(5000, 200, 64, 4, mtrx).to(device)
    }

    dl_probas = {}
    criterion = nn.CrossEntropyLoss(weight=cw_t)
    
    for name, m in dl_models.items():
        print(f"Entrenando {name} (Arquitectura Limpia)...")
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        m = train_dl_model(m, tr_loader, vl_loader, criterion, opt, device)
        m.eval()
        with torch.no_grad():
            dl_probas[name] = torch.softmax(m(torch.tensor(X_ts_seq, dtype=torch.long).to(device)), dim=1).cpu().numpy()

    print("\n" + "="*40)
    print(" RENDIMIENTO INDIVIDUAL DE MODELOS ")
    print("="*40)

    for name, probas in cl_probas.items():
        preds = np.argmax(probas, axis=1)
        score = f1_score(y_ts, preds, average='macro')
        print(f"| {name.ljust(15)} | Macro-F1: {score:.4f} |")

    for name, probas in dl_probas.items():
        preds = np.argmax(probas, axis=1)
        score = f1_score(y_ts, preds, average='macro')
        print(f"| {name.ljust(15)} | Macro-F1: {score:.4f} |")

    final_probas = (cl_probas['xgboost'] * 2.0 + cl_probas['naive_bayes'] * 1.0 + dl_probas['cnn'] * 1.0 + dl_probas['lstm'] * 1.0)
    preds = np.argmax(final_probas, axis=1)

    print("\n" + "="*40)
    print(f"MACRO-F1 FINAL ENSAMBLE: {f1_score(y_ts, preds, average='macro'):.4f}")
    print("="*40)
    print(classification_report(y_ts, preds, target_names=enc.classes_))

if __name__ == "__main__":
    run_full_pipeline()