import torch
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .validate import validate
from .early_stopping import EarlyStopping
from .logger import log_metrics
from transformers import AutoTokenizer

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience, save_path, log_dir=None):
    early_stopper = EarlyStopping(patience=patience, checkpoint_path=f"{save_path}/best_model.pt")
    
    # If you want to limit training samples, slice the train_loader's dataset
    # if len(train_loader.dataset) > 100:  # Only if dataset is larger than 100
    #     train_loader.dataset.dataframe = train_loader.dataset.dataframe.iloc[:100]
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        # Add tqdm progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            images, texts, labels = batch
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, texts)  # [batch_size]
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.sigmoid(outputs) > 0.5  # binary classification
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate training metrics
        train_metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted'),
            'recall': recall_score(true_labels, predictions, average='weighted'),
            'f1': f1_score(true_labels, predictions, average='weighted')
        }
        
        # Validation phase
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Logging
        logging.info(f"Epoch {epoch+1}/{epochs}")
        logging.info(f"Train Loss: {train_metrics['loss']:.4f}, Train F1: {train_metrics['f1']:.4f}")
        logging.info(f"Val Loss: {val_metrics['loss']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        log_metrics(epoch, train_metrics, val_metrics, log_dir)
        
        early_stopper(val_metrics['f1'], model)
        if early_stopper.early_stop:
            logging.info("Early stopping triggered.")
            break
    
    return model
