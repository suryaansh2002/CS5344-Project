import os
import sys
import shutil
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import torch
import pandas as pd
import logging
from torch.utils.data import DataLoader
from datasets.multimodal_dataset import MultiModalDataset, collate_fn
from models.efficientnet_bilstm import MultiModalModel
from training.train import train_model
from training.evaluate import evaluate
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from torchvision import transforms

# ------------------ Config ------------------
DATA_PATH = "DATA/balanced_dataset.csv"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 3
LR = 1e-3
SAVE_PATH = "checkpoints/bilstm_efficientnet/"
LOG_DIR = "logs/bilstm_efficientnet/"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu") 
# Function to clear the directory
def clear_log_dir(log_dir):
    # Check if the directory exists
    if os.path.exists(log_dir):
        # Remove all files inside the directory
        for filename in os.listdir(log_dir):
            file_path = os.path.join(log_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete folder and its contents
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

os.makedirs("checkpoints", exist_ok=True)
clear_log_dir(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)


# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.info("Starting script...")
logging.info(f"Using device: {DEVICE}")

# ------------------ TOKENIZER AND TRANSFORMS ------------------

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
def text_pipeline(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="pt")

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
logging.info("Tokenizer and image transform initialized.")

# ------------------ DATA LOADERS ------------------

df = pd.read_csv(DATA_PATH)  
train_df, temp_df = train_test_split(df, test_size=0.6, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df = train_df.head(100)  
val_df = val_df.head(20)     
test_df = test_df.head(20)   

train_dataset = MultiModalDataset(train_df, text_pipeline, image_transform, image_size=IMAGE_SIZE, tokenizer=tokenizer)
val_dataset = MultiModalDataset(val_df, text_pipeline, image_transform, image_size=IMAGE_SIZE, tokenizer=tokenizer)
test_dataset = MultiModalDataset(test_df, text_pipeline, image_transform, image_size=IMAGE_SIZE, tokenizer=tokenizer)

pad_token_id = tokenizer.pad_token_id

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=lambda x: collate_fn(x, pad_token_id=pad_token_id)
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=lambda x: collate_fn(x, pad_token_id=pad_token_id)
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=lambda x: collate_fn(x, pad_token_id=pad_token_id)
)
logging.info("Data loaders created.")

# --- MODEL, LOSS & OPTIMIZER ---
model = MultiModalModel(tokenizer=tokenizer).to(DEVICE)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
logging.info("Model initialized.")

# ------------------ TRAIN ------------------
logging.info("Starting training...")
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=DEVICE,
    epochs=EPOCHS,
    patience=PATIENCE,
    save_path=SAVE_PATH,
    log_dir=LOG_DIR
)

# --- EVALUATION (TESTING) ---
logging.info("Evaluating the model on the test set...")
acc, f1, precision, recall, report = evaluate(model, test_loader, DEVICE)
logging.info(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
logging.info(f"Classification Report:\n{report}")
