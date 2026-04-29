import re
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TumorRowlandPreprocessor:
    def __init__(self):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
        self.medical_noise = {'received', 'fresh', 'frozen', 'section', 'patient', 'date', 'page', 'copy', 'material', 'examination'}
        self.stop_words.update(self.medical_noise)
        self.lemmatizer = WordNetLemmatizer()

    def clean_for_baseline(self, text):
        text = str(text).lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = re.sub(r'\d+', '[NUM]', text)
        tokens = [self.lemmatizer.lemmatize(t) for t in text.split() if t not in self.stop_words]
        return " ".join(tokens)

    def clean_for_trees(self, text):
        text = str(text).lower()
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        tokens = [self.lemmatizer.lemmatize(t) for t in text.split() if t not in self.medical_noise]
        return " ".join(tokens)

    def clean_for_sequence_models(self, text):
        text = str(text).lower()
        text = re.sub(r'([.,!?()])', r' \1 ', text)
        return text

def augment_organ_aware(texts, labels, generator):
    df = pd.DataFrame({'text': texts, 'label': labels})
    max_size = df['label'].value_counts().max()
    new_texts = []
    new_labels = []
    
    seeds = {
        0: [
            "pathology report: 1cm mass. negative for vascular invasion. no evidence of metastasis. T1.",
            "clinical note: tumor confined to mucosa. unremarkable surrounding tissue. not identified in lymph. T1."
        ],
        3: [
            "URGENT colon path: massv infiltrtn breaching visceral peritoneum. T4! severe organ invsn.",
            "lung biopsy: tumor directly invades mediastinum and pleura. extensive T4 malignancy.",
            "breast oncology: 9cm lesion with direct extension to chest wall and skin ulceration. T4."
        ]
    }
    
    for class_id in [3, 0, 2]:
        class_df = df[df['label'] == class_id]
        deficit = int((max_size - len(class_df)) * 1.25)
        
        if deficit > 0:
            print(f"   -> Generando {deficit} informes SOTA (Organ-Aware) para la clase {class_id}...")
            class_seeds = seeds.get(class_id, ["tumor pathology note:"])
            
            batch_size = 16
            for i in range(0, deficit, batch_size):
                current_batch_size = min(batch_size, deficit - i)
                prompts = [class_seeds[j % len(class_seeds)] for j in range(current_batch_size)]
                
                out = generator(prompts, max_length=150, do_sample=True, top_k=40, top_p=0.92, temperature=0.95, repetition_penalty=1.1)
                
                for seq in out:
                    generated = seq[0]['generated_text'].strip()
                    new_texts.append(generated)
                new_labels.extend([class_id] * current_batch_size)
                
    # FIX: Usar np.concatenate para unir los datos originales (arrays) con los nuevos (listas)
    full_texts = np.concatenate([texts, np.array(new_texts)])
    full_labels = np.concatenate([labels, np.array(new_labels)])
    
    return full_texts, full_labels

class CancerDatasetTTA(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=3000, is_test=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        
        if self.is_test:
            words = text.split()
            v1 = text
            v2 = " ".join(words[50:]) if len(words) > 100 else text
            v3 = " ".join(words[:-50]) if len(words) > 100 else text
            
            encodings = self.tokenizer([v1, v2, v3], truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
            return encodings, torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors="pt")
            item = {key: val.squeeze(0) for key, val in encoding.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item