import os
import logging
import torch
from torch.utils.data import DataLoader
from datasets import MultiModalDataset, collate_fn
from models import MultiModalModel
from training.train import train_model
from training.evaluate import evaluate
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import transforms

# ------------------ Config ------------------
DATA_PATH = "DATA/balanced_dataset.csv"
IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 3
LR = 1e-3
SAVE_PATH = "checkpoints/mobilenet_roberta.pt"
LOG_DIR = "logs/mobilenet_roberta"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu") 
logging.info(f"Using device: {DEVICE}")

# ------------------ LOGGING ------------------

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


# --- TEXT TOKENIZER & TRANSFORM ---
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
def text_pipeline(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="pt")

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
logging.info("Tokenizer and image transform initialized.")

# --- DATA LOADERS ---
df = pd.read_csv(DATA_PATH)  
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = MultiModalDataset(train_df, text_pipeline, image_transform, image_size=IMAGE_SIZE, tokenizer=tokenizer)
val_dataset = MultiModalDataset(val_df, text_pipeline, image_transform, image_size=IMAGE_SIZE, tokenizer=tokenizer)
test_dataset = MultiModalDataset(test_df, text_pipeline, image_transform, image_size=IMAGE_SIZE, tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
logging.info("Data loaders created.")

# --- MODEL, LOSS & OPTIMIZER ---

model = MultiModalModel(alpha=0.5).to(DEVICE)  # Update with your model configuration
logging.info("Model initialized.")

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

logging.info("Loss function and optimizer initialized.")

# --- TRAINING ---
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
