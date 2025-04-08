import torch
import logging

class EarlyStopping:
    def __init__(self, patience=3, checkpoint_path = None):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = checkpoint_path

    def __call__(self, score, model):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            logging.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)
