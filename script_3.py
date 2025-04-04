import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import wandb

wandb.login()

wandb.init(
    project="mmhs150k-multimodal",
    name="efficientnet-lstm",  # experiment name
    config={
        "epochs": 50,
        "batch_size": 2,
        "lr": 1e-3,
        "embedding_dim": 128,
        "hidden_dim": 128,
        "max_length": 100,
        "image_size": 224,
        "model": "EfficientNetB0 + BiLSTM"
    }
)

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- LOAD DATA ---
df = pd.read_csv("balanced_dataset.csv")
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
logging.info("Loaded dataset.")

# --- TEXT TOKENIZER & VOCAB ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def text_pipeline(text):
    # Use Hugging Face tokenizer for text processing
    return tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="pt")

# --- IMAGE TRANSFORMS ---
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# --- CUSTOM DATASET ---
class MultiModalDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return 100
        # return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert("RGB")
            image = image_transform(image)
        except:
            image = torch.zeros(3, 224, 224)

        text = text_pipeline(row['cleaned_text'])["input_ids"].squeeze(0)
        label = torch.tensor(row['binary_label'], dtype=torch.float32)
        return image, text, label

def collate_fn(batch):
    images, texts, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    # Padding text sequences manually as Hugging Face tokenizer handles padding
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id)
    return images, texts_padded, labels

train_loader = DataLoader(MultiModalDataset(train_df), batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(MultiModalDataset(val_df), batch_size=2, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(MultiModalDataset(test_df), batch_size=2, shuffle=False, collate_fn=collate_fn)
logging.info("Data loaders created.")

# --- MODEL ---
class MultiModalModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super(MultiModalModel, self).__init__()

        # Image model
        self.cnn = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        logging.info("Loaded EfficientNet-B0 model.")
        # logging.info(self.cnn)
        self.cnn.classifier = nn.Identity()
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.img_fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Text model
        self.embedding = nn.Embedding(tokenizer.vocab_size, 128, padding_idx=tokenizer.pad_token_id)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True, bidirectional=True)
        self.text_fc = nn.Sequential(
            nn.Linear(2 * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Combined
        self.fc = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        logging.info('Entire model loaded.')

    def forward(self, img, text):
        img_feat = self.img_fc(self.cnn(img))
        text_emb = self.embedding(text)
        _, (hidden, _) = self.lstm(text_emb)
        text_feat = self.text_fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
        combined = torch.cat((img_feat, text_feat), dim=1)
        return self.fc(combined).squeeze(1)

model = MultiModalModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- TRAINING FUNCTION ---
def train_epoch(model, dataloader):
    logging.info("Training epoch...")
    model.train()
    running_loss = 0
    for imgs, texts, labels in dataloader:
        imgs, texts, labels = imgs.to(device), texts.to(device), labels.to(device)
        optimizer.zero_grad()
        # logging.info('Model forward pass...')
        outputs = model(imgs, texts)
        # logging.info('Model forward pass complete.')
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# --- EVALUATION ---
def evaluate(model, dataloader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for imgs, texts, labels in dataloader:
            imgs, texts = imgs.to(device), texts.to(device)
            outputs = model(imgs, texts)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return f1, precision, recall, accuracy

# --- EARLY STOPPING CONFIG ---
best_f1 = 0
patience = 3
counter = 0
best_model_path = "best_model.pth"

# --- TRAIN LOOP WITH EARLY STOPPING ---
logging.info("Starting training with early stopping...")

for epoch in range(1, 50):  # set a high max epoch; early stopping will break earlier
    loss = train_epoch(model, train_loader)

    val_f1, val_precision, val_recall, val_accuracy = evaluate(model, val_loader)
    logging.info(f"Epoch {epoch} - Loss: {loss:.4f} | Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
    wandb.log({
        "epoch": epoch,
        "train_loss": loss,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_accuracy": val_accuracy
    })
    # Check for improvement
    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"New best F1: {best_f1:.4f}. Saving model.")
    else:
        counter += 1
        logging.info(f"No improvement. Early stopping counter: {counter}/{patience}")
        if counter >= patience:
            logging.info("Early stopping triggered.")
            break

# --- LOAD BEST MODEL BEFORE TESTING ---
model.load_state_dict(torch.load(best_model_path))
model.eval()

# --- FINAL TEST ---
test_f1, test_precision, test_recall, test_accuracy = evaluate(model, test_loader)
logging.info(f"Test Results -> Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
wandb.log({
    "test_f1": test_f1,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_accuracy": test_accuracy
})
