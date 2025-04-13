import os
import sys
import shutil
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import logging
import torch
from torch.utils.data import DataLoader
from datasets.multimodal_dataset import MultiModalDataset, collate_fn
from models.mobilenet_roberta import MultiModalModel
from training.train import train_model
from training.evaluate import evaluate
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision import transforms
from plot_alpha_results import plotAlphaResults
from plot_mobilenet_roberta_final_alpha_metrics import plotroBERTaMobileNet

# ------------------ CONFIG ------------------
DATA_PATH = "DATA/balanced_dataset.csv"
IMAGE_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 3
LR = 1e-3
SAVE_PATH = "checkpoints/mobilenet_roberta/"
LOG_DIR = "logs/mobilenet_roberta/"
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
logging.info(f"Using device: {DEVICE}")
logging.info("Starting script...")

# --- TEXT TOKENIZER & TRANSFORM ---
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
def text_pipeline(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=100, return_tensors="pt")

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])
logging.info("Tokenizer and image transform initialized.")

# --- LOAD DATA ---
df = pd.read_csv(DATA_PATH)
train_df, temp_df = train_test_split(df, test_size=0.6, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# change to whole dataset
train_df = train_df.head(100)
val_df = val_df.head(20)
test_df = test_df.head(20)

# --- DATASETS & LOADERS ---
pad_token_id = tokenizer.pad_token_id

def get_loader(df):
    dataset = MultiModalDataset(df, text_pipeline, image_transform, image_size=IMAGE_SIZE, tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_token_id=pad_token_id))

train_loader = get_loader(train_df)
val_loader = get_loader(val_df)
test_loader = get_loader(test_df)

logging.info("Data loaders created.")

# --- GRID SEARCH ON ALPHA ---
alpha_results = []
best_alpha = 0.5

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    logging.info(f"\n--- Running for alpha={alpha} ---")

    model = MultiModalModel(alpha=alpha).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.BCEWithLogitsLoss()

    current_log_dir = os.path.join(LOG_DIR, f"alpha_{alpha}")
    current_save_path = os.path.join(SAVE_PATH, f"alpha_{alpha}")
    os.makedirs(current_log_dir, exist_ok=True)
    os.makedirs(current_save_path, exist_ok=True)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        epochs=EPOCHS,
        patience=PATIENCE,
        save_path=current_save_path,
        log_dir=current_log_dir
    )

    acc, f1, precision, recall, report = evaluate(model, val_loader, DEVICE)
    logging.info(f"alpha={alpha} | Val Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    alpha_results.append({
        'alpha': alpha,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    })

# --- SELECT BEST ALPHA ---
best_result = max(alpha_results, key=lambda x: x['f1'])
best_alpha = best_result['alpha']
logging.info(f"\nBest alpha on validation set: {best_alpha} (F1: {best_result['f1']:.4f})")

# --- RETRAIN ON TRAIN + VAL AND TEST ---
combined_df = pd.concat([train_df, val_df]).reset_index(drop=True)
combined_loader = get_loader(combined_df)

final_model = MultiModalModel(alpha=best_alpha).to(DEVICE)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=LR)
final_criterion = torch.nn.BCEWithLogitsLoss()

final_log_dir = os.path.join(LOG_DIR, "final_model")
final_save_path = os.path.join(SAVE_PATH, "final_model")
os.makedirs(final_log_dir, exist_ok=True)
os.makedirs(final_save_path, exist_ok=True)

train_model(
    model=final_model,
    train_loader=combined_loader,
    val_loader=val_loader,
    criterion=final_criterion,
    optimizer=final_optimizer,
    device=DEVICE,
    epochs=EPOCHS,
    patience=PATIENCE,
    save_path=final_save_path,
    log_dir=final_log_dir
)

acc, f1, precision, recall, report = evaluate(final_model, test_loader, DEVICE)
logging.info(f"\nFinal Test Metrics (after alpha tuning):\nAccuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
logging.info(f"Classification Report:\n{report}")

plotAlphaResults()
plotroBERTaMobileNet()
