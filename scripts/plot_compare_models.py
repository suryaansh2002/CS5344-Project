import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
bilstm_path = "logs/bilstm_efficientnet/metrics.csv"
mobilenet_path = "logs/mobilenet_roberta/final_model/metrics.csv"
save_path = "plots/model_comparison_grid.png"
os.makedirs("plots", exist_ok=True)

# --- LOAD METRICS ---
df_bilstm = pd.read_csv(bilstm_path)
df_mobilenet = pd.read_csv(mobilenet_path)

# --- TRIM TO MINIMUM EPOCHS ---
min_epochs = min(len(df_bilstm), len(df_mobilenet))
df_bilstm = df_bilstm.iloc[:min_epochs]
df_mobilenet = df_mobilenet.iloc[:min_epochs]

# Add model labels
df_bilstm["model"] = "BiLSTM + EfficientNet"
df_mobilenet["model"] = "RoBERTa + MobileNet"

# --- COMBINE DATA ---
combined = pd.concat([df_bilstm, df_mobilenet], ignore_index=True)

# --- METRICS TO PLOT ---
metrics = [
    ("train_loss", "Training Loss"),
    ("val_loss", "Validation Loss"),
    ("train_accuracy", "Training Accuracy"),
    ("val_accuracy", "Validation Accuracy"),
    ("train_f1", "Training F1 Score"),
    ("val_f1", "Validation F1 Score")
]

sns.set_theme(style='darkgrid', palette='pastel', font='sans-serif', font_scale=1, rc=None)

# --- CREATE PLOTS ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, (metric, title) in zip(axes, metrics):
    sns.lineplot(data=combined, x="epoch", y=metric, hue="model", marker="o", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("")
    ax.legend().set_title("")  # Hide legend title for clean look

# --- Adjust layout ---
plt.tight_layout()
plt.suptitle("Model Comparison over Epochs", fontsize=16, y=1.02)
plt.subplots_adjust(top=0.92)

# --- Save & Done ---
plt.savefig(save_path)
plt.show()
print(f"Combined plot saved to: {save_path}")
