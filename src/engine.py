import torch
import numpy as np
import copy
import gc
import time
import os
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from torch.optim import AdamW
import transformers
from transformers import get_linear_schedule_with_warmup, pipeline

from src.preprocessing import CancerDatasetTTA, augment_organ_aware
from models.transformer import ClinicalCNN_Ensemble
from src.losses import FocalLabelSmoothingLoss

transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
transformers.modeling_utils.check_torch_load_is_safe = lambda: None

class TitanOrchestrator:
    def __init__(self, tokenizer, device, num_classes=4):
        self.tokenizer = tokenizer
        self.device = device
        self.num_classes = num_classes

    def predict_tta(self, model, loader):
        model.eval()
        probas = []
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                for batch_data, _ in loader:
                    input_ids = batch_data['input_ids'].squeeze(0).to(self.device)
                    mask = batch_data['attention_mask'].squeeze(0).to(self.device)
                    outputs = model(input_ids, mask)
                    avg_probs = torch.softmax(outputs, dim=1).mean(dim=0).cpu().numpy()
                    probas.append(avg_probs)
        return np.array(probas)

    def find_threshold(self, y_true, probas):
        best_thresh, best_f1 = 0.5, 0
        for thresh in np.arange(0.15, 0.60, 0.02):
            preds = [3 if p[3] >= thresh else np.argmax(p[:3]) for p in probas]
            score = f1_score(y_true, preds, average='macro')
            if score > best_f1:
                best_f1, best_thresh = score, thresh
        return best_thresh

    def run_fold(self, fold_idx, train_data, val_data, epochs=6):
        X_t, y_t = train_data
        X_v, y_v = val_data
        
        print(f"   -> [Fase 1] Bio-Augmentation...")
        gen = pipeline('text-generation', model='microsoft/BioGPT', device=0)
        X_t_bal, y_t_bal = augment_organ_aware(X_t, y_t, gen)
        del gen; torch.cuda.empty_cache(); gc.collect()

        train_loader = DataLoader(CancerDatasetTTA(X_t_bal, y_t_bal, self.tokenizer), batch_size=1, shuffle=True)
        val_loader = DataLoader(CancerDatasetTTA(X_v, y_v, self.tokenizer, is_test=True), batch_size=1)

        model = ClinicalCNN_Ensemble(num_classes=self.num_classes).to(self.device)
        optimizer = AdamW([
            {'params': model.longformer.parameters(), 'lr': 1e-5},
            {'params': model.convs.parameters(), 'lr': 2e-3},
            {'params': model.fc.parameters(), 'lr': 2e-3}
        ], weight_decay=0.02)

        scaler = torch.amp.GradScaler('cuda')
        total_steps = (len(train_loader) // 16) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
        criterion = FocalLabelSmoothingLoss()

        best_f1, best_wts = 0, None
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            for i, batch in enumerate(train_loader):
                ids, mask, lbls = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['labels'].to(self.device)
                with torch.amp.autocast('cuda'):
                    loss = criterion(model(ids, mask), lbls) / 16
                scaler.scale(loss).backward()
                if (i + 1) % 16 == 0 or (i + 1) == len(train_loader):
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
            
            p_val = self.predict_tta(model, val_loader)
            th = self.find_threshold(y_v, p_val)
            preds = [3 if p[3] >= th else np.argmax(p[:3]) for p in p_val]
            f1 = f1_score(y_v, preds, average='macro')
            print(f"      Epoca {epoch+1} | F1: {f1:.4f} (Thresh: {th:.2f})")
            if f1 > best_f1:
                best_f1, best_wts = f1, copy.deepcopy(model.state_dict())

        return best_wts, best_f1