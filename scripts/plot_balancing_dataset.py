import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

PLOT_SAVE_DIR = "plots"
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
PLOT_SAVE_PATH = os.path.join(PLOT_SAVE_DIR, "balancing_dataset.png")

# Set style
sns.set_theme(style='darkgrid', palette='pastel', font='sans-serif', font_scale=1, rc=None)

# Before balancing
before_counts = pd.DataFrame({
    'Label': ['NotHate', 'Hate'],
    'Count': [116790, 33033],
    'Stage': 'Before Balancing'
})

# After balancing
after_counts = pd.DataFrame({
    'Label': ['NotHate', 'Hate'],
    'Count': [75000, 75000],
    'Stage': 'After Balancing'
})

# Combine both for plotting
combined_df = pd.concat([before_counts, after_counts])

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=combined_df, x='Label', y='Count', hue='Stage')
plt.title('Class Distribution Before and After Balancing')
plt.ylabel('Number of Samples')
plt.xlabel('Class Label')
plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH, dpi=300)
plt.show()


