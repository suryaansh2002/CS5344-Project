import os
import pandas as pd
import logging

def log_metrics(epoch, train_metrics, val_metrics, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "metrics.csv")

    log_data = {
        "epoch": epoch + 1,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()}
    }

    df = pd.DataFrame([log_data])

    # Check if file exists to determine if we need to write the header
    file_exists = os.path.exists(log_path)
    
    # Write to CSV with header only if file doesn't exist
    df.to_csv(log_path, mode='a', header=not file_exists, index=False)
    logging.info('Training metrics saved to csv.')
