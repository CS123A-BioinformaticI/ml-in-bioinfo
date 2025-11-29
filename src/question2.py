import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load dataset
df = pd.read_csv("pairs_with_features.csv")

# Scatter plot
plt.figure(figsize=(8, 6))
sns.regplot(x='align_score', y='percent_identity', data=df)

plt.xlabel('Alignment Score')
plt.ylabel('Percent Identity (%)')
plt.title('Correlation Between Alignment Score and Percent Identity')
plt.show()

# # Pearson correlation
# corr, p_value = pearsonr(df['align_score'], df['percent_identity'])
# print(f"Pearson correlation: {corr:.3f}, p-value: {p_value:.5f}")
