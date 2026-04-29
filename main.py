import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.engine import TitanOrchestrator
from src.preprocessing import CancerDatasetTTA
from models.transformer import ClinicalCNN_Ensemble

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv('data/tcga_simple_train.csv')
    df_dev = pd.read_csv('data/tcga_simple_dev_shuffled.csv')
    
    enc = LabelEncoder()
    df['label'] = enc.fit_transform(df['t'])
    X, y = df['text'].values, df['label'].values

    tokenizer = AutoTokenizer.from_pretrained("yikuan8/Clinical-Longformer")
    orchestrator = TitanOrchestrator(tokenizer, device)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    test_loader = DataLoader(CancerDatasetTTA(df_dev['text'].tolist(), [0]*len(df_dev), tokenizer, is_test=True), batch_size=1)
    ensemble_probas = np.zeros((len(df_dev), 4))
    oof_probas_full = np.zeros((len(X), 4))
    os.makedirs('models/definitive', exist_ok=True)

    for fold, (t_idx, v_idx) in enumerate(skf.split(X, y)):
        print(f"\n>>> INICIANDO FOLD {fold+1}/3")
        wts, f1 = orchestrator.run_fold(fold, (X[t_idx], y[t_idx]), (X[v_idx], y[v_idx]))
        
        model = ClinicalCNN_Ensemble().to(device)
        model.load_state_dict(wts)
        
        val_loader = DataLoader(CancerDatasetTTA(X[v_idx].tolist(), y[v_idx].tolist(), tokenizer, is_test=True), batch_size=1)
        oof_probas_full[v_idx] = orchestrator.predict_tta(model, val_loader)
        
        ensemble_probas += orchestrator.predict_tta(model, test_loader) / 3.0
        torch.save(wts, f'models/definitive/fold_{fold+1}.pth')

    best_thresh = orchestrator.find_threshold(y, oof_probas_full)
    print(f"\n[CALIBRACION] Umbral T4 optimo global: {best_thresh:.2f}")

    preds = [3 if p[3] >= best_thresh else np.argmax(p[:3]) for p in ensemble_probas]
    df_dev['t'] = enc.inverse_transform(preds)
    df_dev.to_csv('submission_final_sota.csv', index=False)

if __name__ == "__main__":
    main()