import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from datetime import datetime
import numpy as np

class Tokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, texts):
        idx = 2
        for text in texts:
            for token in re.findall(r'\w+', text.lower()):
                if token not in self.word2idx:
                    self.word2idx[token] = idx
                    idx += 1

    def encode(self, text, max_len):
        tokens = re.findall(r'\w+', text.lower())
        ids = [self.word2idx.get(t, 1) for t in tokens]
        return ids[:max_len] + [0] * max(0, max_len - len(ids))

class CategoricalEncoder:
    def __init__(self):
        self.label2idx = {}

    def fit(self, values):
        self.label2idx = {v: i for i, v in enumerate(sorted(set(values)))}

    def encode(self, value):
        return self.label2idx.get(value, 0)

    def vocab_size(self):
        return len(self.label2idx)

def extract_time_features(timestamp_str):
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    return torch.tensor([
        dt.hour / 23.0,
        dt.weekday() / 6.0,
        dt.timetuple().tm_yday / 365.0
    ], dtype=torch.float32)

class LogDataset(Dataset):
    def __init__(self, logs, tokenizer, level_enc, func_enc, max_len):
        self.logs = logs
        self.tokenizer = tokenizer
        self.level_enc = level_enc
        self.func_enc = func_enc
        self.max_len = max_len

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        log = self.logs[idx]

        msg_ids = self.tokenizer.encode(log["message"], self.max_len)
        msg_tensor = torch.tensor(msg_ids, dtype=torch.long)

        level_idx = self.level_enc.encode(log["level"])
        func_idx = self.func_enc.encode(log["function"])
        meta = extract_time_features(log["timestamp"])

        label = torch.tensor(level_idx, dtype=torch.long)
        return msg_tensor, func_idx, meta, label

class LogClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, func_vocab_size, meta_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.func_embed = nn.Embedding(func_vocab_size, 8)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 8 + meta_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, msg_seq, func_ids, meta):
        embedded = self.embedding(msg_seq)               # (B, L, E)
        _, (h_n, _) = self.lstm(embedded)                # h_n: (1, B, H)
        h_last = h_n.squeeze(0)                          # (B, H)

        func_vec = self.func_embed(func_ids)             # (B, 8)
        combined = torch.cat([h_last, func_vec, meta], dim=1)  # (B, H+8+M)
        return self.fc(combined)

def train_model(model, dataloader, num_epochs=5, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for msg, func_id, meta, label in dataloader:
            optimizer.zero_grad()
            output = model(msg, func_id, meta)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def preprocess_log(log, tokenizer, func_enc, max_len):
    msg_ids = tokenizer.encode(log["message"], max_len)
    msg_tensor = torch.tensor([msg_ids], dtype=torch.long)  # shape (1, L)

    func_idx = func_enc.encode(log["function"])
    func_tensor = torch.tensor([func_idx], dtype=torch.long)  # shape (1,)

    meta_tensor = extract_time_features(log["timestamp"]).unsqueeze(0)  # shape (1, 3)

    return msg_tensor, func_tensor, meta_tensor

def predict_log(model, log_entry, tokenizer, func_enc, level_enc, max_len):
    model.eval()
    with torch.no_grad():
        msg, func_id, meta = preprocess_log(log_entry, tokenizer, func_enc, max_len)
        logits = model(msg, func_id, meta)         # (1, num_classes)
        probs = torch.softmax(logits, dim=1)       # (1, num_classes)
        pred_class = torch.argmax(probs, dim=1).item()
        label = list(level_enc.label2idx.keys())[pred_class]
        return label, probs.squeeze().tolist()

if __name__ == "__main__":
    logs = [
        {"timestamp": "2024-03-21 10:15:32", "level": "ERROR", "function": "connect_to_db", "message": "Failed to connect to database"},
        {"timestamp": "2024-03-21 10:17:01", "level": "INFO", "function": "start_service", "message": "Service started successfully"},
        {"timestamp": "2024-03-21 10:17:05", "level": "WARN", "function": "retry_login", "message": "Retrying login after timeout"},
    ]

    tokenizer = Tokenizer()
    tokenizer.fit([l["message"] for l in logs])

    level_enc = CategoricalEncoder()
    level_enc.fit([l["level"] for l in logs])

    func_enc = CategoricalEncoder()
    func_enc.fit([l["function"] for l in logs])

    dataset = LogDataset(logs, tokenizer, level_enc, func_enc, max_len=10)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = LogClassifier(
        vocab_size=len(tokenizer.word2idx),
        embed_dim=50,
        hidden_dim=64,
        func_vocab_size=func_enc.vocab_size(),
        meta_dim=3,
        num_classes=level_enc.vocab_size()
    )

    train_model(model, loader, num_epochs=100)

    input()

    new_log = {
        "timestamp": "2024-03-21 11:32:00",
        "level": "UNKNOWN",  # not used here, but okay to include
        "function": "start_service",
        "message": "System startup complete"
    }

    pred_label, prob_scores = predict_log(
        model, new_log, tokenizer, func_enc, level_enc, max_len=10
    )

    print(f"Predicted label: {pred_label}")
    print(f"Class probabilities: {prob_scores}")


