from datasets.multimodal_dataset import MultiModalDataset, collate_fn
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from itertools import product
from tqdm import tqdm

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
logging.info(f"Using device: {device}")

# --- LOAD DATA ---
df = pd.read_csv("balanced_dataset.csv")
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# --- IMAGE TRANSFORMS ---
image_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])

# --- TEXT TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")

def text_pipeline(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="pt")

# --- CUSTOM DATASET AND DATALOADERS ---
train_dataset = MultiModalDataset(train_df, tokenizer, image_transform)
logging.info("Custom dataset created.")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))
val_loader = DataLoader(MultiModalDataset(val_df, tokenizer, image_transform), batch_size=32, shuffle=False,
                        collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))
test_loader = DataLoader(MultiModalDataset(test_df, tokenizer, image_transform), batch_size=32, shuffle=False,
                         collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))
logging.info("Data loaders created.")

# --- IMAGE MODEL ---
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 256)
        )
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.model(x)

# --- TEXT MODEL ---
class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
        # logging.info(self.model)
        self.fc = nn.Linear(768, 256)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        output = self.model(x).last_hidden_state[:, 0, :]
        return self.fc(output)

# --- MULTIMODAL MODEL ---
class MultiModalModel(nn.Module):
    def __init__(self, alpha=0.5):
        super(MultiModalModel, self).__init__()
        self.image_model = ImageModel()
        self.text_model = TextModel()
        self.alpha = alpha
        self.fc = nn.Linear(256, 1)
    
    def forward(self, img, text):
        img_feat = self.image_model(img)
        text_feat = self.text_model(text)
        combined = self.alpha * img_feat + (1-self.alpha) * text_feat
        return self.fc(combined).squeeze(1)

# --- TRAIN FUNCTION WITH EARLY STOPPING ---
def train_model(model, dataloader, optimizer, criterion, val_loader, patience=3, epochs=50, save_path="best_model_roberta_mobilenet.pth"):
    model.train()
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        all_preds, all_labels = [], []
        for imgs, texts, labels in tqdm(dataloader):
            imgs, texts, labels = imgs.to(device), texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
        
        train_epoch_f1 = f1_score(all_labels, all_preds)
        train_accuracy_score = np.mean(np.array(all_preds) == np.array(all_labels))
        train_precision = precision_score(all_labels, all_preds)
        train_recall = recall_score(all_labels, all_preds)
        
        logging.info(f"Epoch {epoch+1}: Loss = {running_loss / len(dataloader):.4f}, Train F1 = {train_epoch_f1:.4f}, Train Accuracy = {train_accuracy_score:.4f}, Train Precision = {train_precision:.4f}, Train Recall = {train_recall:.4f}")
        val_model(model, val_loader)

        if train_epoch_f1 > best_f1:
            best_f1 = train_epoch_f1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            logging.info(f"No improvement. Early stopping counter: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            logging.info("Early stopping triggered")
            break

# --- FINAL TESTING ---
def test_model(model, dataloader, load_path="best_model_roberta_mobilenet.pth"):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, texts, labels in dataloader:
            imgs, texts = imgs.to(device), texts.to(device)
            outputs = torch.sigmoid(model(imgs, texts)).cpu().numpy()
            preds = (outputs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    test_f1 = f1_score(all_labels, all_preds)
    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    test_precision = precision_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    logging.info(f"Test F1 Score: {test_f1:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    
def val_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, texts, labels in dataloader:
            imgs, texts = imgs.to(device), texts.to(device)
            outputs = torch.sigmoid(model(imgs, texts)).cpu().numpy()
            preds = (outputs > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    val_f1 = f1_score(all_labels, all_preds)
    val_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    val_precision = precision_score(all_labels, all_preds)
    val_recall = recall_score(all_labels, all_preds)
    logging.info(f"Validation F1 Score: {val_f1:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")

# --- RUN TRAINING AND TESTING ---
model = MultiModalModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
logging.info("Starting training with early stopping...")
train_model(model, train_loader, optimizer, criterion, val_loader)
test_model(model, test_loader)