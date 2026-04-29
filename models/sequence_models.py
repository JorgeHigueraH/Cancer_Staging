import torch
import torch.nn as nn
import numpy as np

class TumorRowlandCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, matrix=None, trainable=True):
        super(TumorRowlandCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if matrix is not None and np.any(matrix):
            self.embedding.weight.data.copy_(torch.from_numpy(matrix))
        self.embedding.weight.requires_grad = trainable

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=k)
            for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        features = []
        for conv in self.convs:
            feat = torch.relu(conv(x))
            feat = torch.max(feat, dim=2)[0]
            features.append(feat)
        out = torch.cat(features, dim=1)
        return self.fc(self.dropout(out))

class TumorRowlandRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, matrix=None, trainable=True):
        super(TumorRowlandRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if matrix is not None and np.any(matrix):
            self.embedding.weight.data.copy_(torch.from_numpy(matrix))
        self.embedding.weight.requires_grad = trainable

        self.rnn = nn.RNN(
            input_size=embed_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        
        hidden_concat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        out = self.dropout(hidden_concat)
        return self.fc(out)

class TumorRowlandLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, matrix=None, trainable=True):
        super(TumorRowlandLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if matrix is not None and np.any(matrix):
            self.embedding.weight.data.copy_(torch.from_numpy(matrix))
        self.embedding.weight.requires_grad = trainable

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        pooled = torch.max(out, dim=1)[0]
        return self.fc(self.dropout(pooled))