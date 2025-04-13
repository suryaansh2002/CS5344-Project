import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
test_metrics_path = "logs/test_metrics.csv"
save_path = "plots/test_metrics_graph.png"
os.makedirs("plots", exist_ok=True)

# --- LOAD METRICS ---
df = pd.read_csv(test_metrics_path)  

df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

sns.set_theme(style='darkgrid', palette='pastel', font='sans-serif', font_scale=1, rc=None)

plt.figure(figsize=(8, 6))
plot = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="Set2")

for p in plot.patches:
    plot.annotate(f'{p.get_height():.3f}', 
                  (p.get_x() + p.get_width() / 2., p.get_height()), 
                  ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5),
                  textcoords='offset points')

plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.xlabel("Metric")
plt.legend(title="Model")
plt.tight_layout()

plt.savefig(save_path)
plt.show()
