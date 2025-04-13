import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, texts, labels in dataloader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return {
        "loss": val_loss / len(dataloader),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
