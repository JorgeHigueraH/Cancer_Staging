import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

class TumorRowlandTransformer(nn.Module):
    def __init__(self, model_name, num_classes=4):
        super(TumorRowlandTransformer, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            use_safetensors=True
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

def get_transformer_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

class ClinicalCNN_Ensemble(nn.Module):
    def __init__(self, model_name="yikuan8/Clinical-Longformer", num_classes=4):
        super(ClinicalCNN_Ensemble, self).__init__()
        self.longformer = AutoModel.from_pretrained(model_name)
        self.longformer.gradient_checkpointing_enable()
        
        hidden_size = self.longformer.config.hidden_size
        num_filters = 150
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=num_filters, kernel_size=k)
            for k in [2, 3, 4, 5]
        ])
        
        self.dropouts = nn.ModuleList([nn.Dropout(0.1 + (i * 0.1)) for i in range(5)])
        self.fc = nn.Linear(num_filters * 4, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state.permute(0, 2, 1)
        conv_res = [F.max_pool1d(F.relu(c(x)), F.relu(c(x)).size(2)).squeeze(2) for c in self.convs]
        out = torch.cat(conv_res, 1)
        return torch.mean(torch.stack([self.fc(d(out)) for d in self.dropouts]), dim=0)