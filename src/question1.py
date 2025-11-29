import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("pairs_with_features.csv")

# Convert pairwise dataset to per-sequence dataset
df_A = df[['seqA_file', 'familyA', 'lenA']].rename(
    columns={'seqA_file': 'sequence', 'familyA': 'family', 'lenA': 'length'})
df_B = df[['seqB_file', 'familyB', 'lenB']].rename(
    columns={'seqB_file': 'sequence', 'familyB': 'family', 'lenB': 'length'})

# Merge both into one sequence-level dataframe
seq_df = pd.concat([df_A, df_B], ignore_index=True).drop_duplicates()

print(seq_df.head())

plt.figure(figsize=(9, 6))
sns.boxplot(x='family', y='length', data=seq_df)

plt.title("Sequence Length Distribution Across Protein Families")
plt.xlabel("Protein Family")
plt.ylabel("Sequence Length")
plt.xticks(rotation=45)
plt.show()
