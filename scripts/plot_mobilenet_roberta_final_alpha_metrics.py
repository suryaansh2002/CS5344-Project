import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Config ---
LOG_PATH = "logs/mobilenet_roberta/final_model" 
PLOT_SAVE_DIR = "plots"
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
PLOT_SAVE_PATH = os.path.join(PLOT_SAVE_DIR, "mobilenet_roberta_metrics.png")

def plotroBERTaMobileNet():
    sns.set_theme(style='darkgrid', palette='pastel', font='sans-serif', font_scale=1, rc=None)

    # --- Collect metrics ---
    metrics_file = os.path.join(LOG_PATH, "metrics.csv")

    # Read the CSV file
    df = pd.read_csv(metrics_file)

    # --- Melt into long format for Seaborn ---
    train_melted = df.melt(
        id_vars=["epoch"], 
        value_vars=["train_f1", "train_accuracy", "train_precision", "train_recall"],
        var_name="Metric", 
        value_name="Score"
    )

    val_melted = df.melt(
        id_vars=["epoch"], 
        value_vars=["val_f1", "val_accuracy", "val_precision", "val_recall"],
        var_name="Metric", 
        value_name="Score"
    )

    # --- Plotting ---

    plt.figure(figsize=(14, 8))

    # Plot training metrics
    sns.lineplot(
        data=train_melted, 
        x="epoch", 
        y="Score", 
        hue="Metric", 
        marker="o", 
        linewidth=2.5
    )

    # Plot validation metrics on top of training metrics
    sns.lineplot(
        data=val_melted, 
        x="epoch", 
        y="Score", 
        hue="Metric", 
        marker="*", 
        linewidth=2.5, 
        linestyle="--"
    )

    plt.title("Training & Validation Metrics — MobileNet + roBERTa", fontsize=18, weight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Metric", fontsize=12, title_fontsize=13)
    plt.tight_layout()

    # Save plot
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    plt.show()

# plotroBERTaMobileNet()
