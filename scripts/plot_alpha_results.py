import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Config ---
LOG_BASE_PATH = "logs/mobilenet_roberta"
PLOT_SAVE_DIR = "plots"
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
PLOT_SAVE_PATH = os.path.join(PLOT_SAVE_DIR, "alpha_val_metrics.png")

# --- Collect metrics ---
def plotAlphaResults():
    sns.set_theme(style='darkgrid', palette='pastel', font='sans-serif', font_scale=1, rc=None)
    alpha_results = []

    for folder in sorted(os.listdir(LOG_BASE_PATH)):
        if folder.startswith("alpha_"):
            try:
                alpha = float(folder.split("_")[1])
            except ValueError:
                continue

            metrics_path = os.path.join(LOG_BASE_PATH, folder, "metrics.csv")
            if os.path.exists(metrics_path):
                # Read the CSV file, skipping the repeated headers
                df = pd.read_csv(metrics_path, skiprows=lambda i: i > 0 and i % 2 == 0)
                best_idx = df["val_f1"].idxmax()
                best_row = df.loc[best_idx]

                alpha_results.append({
                    "alpha": alpha,
                    "epoch": int(best_row["epoch"]),
                    "train_f1": best_row["train_f1"],
                    "train_accuracy": best_row["train_accuracy"],
                    "val_f1": best_row["val_f1"],
                    "val_accuracy": best_row["val_accuracy"],
                    "val_precision": best_row["val_precision"],
                    "val_recall": best_row["val_recall"]
                })

    results_df = pd.DataFrame(alpha_results)

    # --- Melt into long format for Seaborn ---
    val_melted = results_df.melt(
        id_vars=["alpha", "epoch"], 
        value_vars=["val_f1", "val_accuracy", "val_precision", "val_recall"],
        var_name="Metric", 
        value_name="Score"
    )

    # --- Plotting ---
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("Set2")

    plt.figure(figsize=(14, 8))
    ax = sns.lineplot(
        data=val_melted, 
        x="alpha", 
        y="Score", 
        hue="Metric", 
        marker="o", 
        palette=palette,
        linewidth=2.5
    )

    # Annotate best epoch
    for i, row in results_df.iterrows():
        ax.annotate(
            f"ep{row['epoch']}", 
            (row["alpha"], row["val_f1"] + 0.005),
            textcoords="offset points", 
            xytext=(0, 8), 
            ha='center', 
            fontsize=10,
            color="dimgray"
        )

    plt.title("Validation Metrics vs Alpha — MobileNet + RoBERTa", fontsize=18, weight='bold')
    plt.xlabel("Alpha (Weight for Image Model)", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(results_df["alpha"].sort_values().unique(), fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Validation Metric", fontsize=12, title_fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    plt.show()
