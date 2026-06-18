import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 110
os.makedirs('figures', exist_ok=True)

# -----------------------------------------------------------------
# 1. LOAD AND CLEAN
# -----------------------------------------------------------------
df = pd.read_csv('customer_churn_dataset-training-master.csv')
print(f"Raw shape: {df.shape}")

# Drop the single fully-empty row (artifact at row 199295)
df = df.dropna().reset_index(drop=True)
df['CustomerID'] = df['CustomerID'].astype(int)
df['Churn'] = df['Churn'].astype(int)
print(f"Shape after dropping null row: {df.shape}")
print(f"Duplicate rows: {df.duplicated().sum()}")
print(f"Remaining missing values: {df.isnull().sum().sum()}")

# Save cleaned dataset for reuse in Task 2 and for Tableau (Task 3)
df.to_csv('churn_cleaned.csv', index=False)

# -----------------------------------------------------------------
# 2. SUMMARY STATISTICS
# -----------------------------------------------------------------
print("\n--- Numeric summary ---")
print(df.describe().T.round(2))

print("\n--- Churn balance ---")
churn_counts = df['Churn'].value_counts()
churn_rate = df['Churn'].mean() * 100
print(churn_counts)
print(f"Overall churn rate: {churn_rate:.2f}%")

# -----------------------------------------------------------------
# 3. CHURN RATE BY CATEGORICAL SEGMENT (used in EDA + recommendations)
# -----------------------------------------------------------------
for col in ['Gender', 'Subscription Type', 'Contract Length']:
    rate = df.groupby(col)['Churn'].mean().mul(100).round(2)
    print(f"\nChurn rate by {col}:\n{rate}")

# Bin Age and Tenure for extra insight
df['Age Group'] = pd.cut(df['Age'], bins=[17,25,35,45,55,65],
                          labels=['18-25','26-35','36-45','46-55','56-65'])
df['Tenure Group'] = pd.cut(df['Tenure'], bins=[0,12,24,36,48,60],
                             labels=['0-12','13-24','25-36','37-48','49-60'])
print("\nChurn rate by Age Group:\n", df.groupby('Age Group')['Churn'].mean().mul(100).round(2))
print("\nChurn rate by Tenure Group:\n", df.groupby('Tenure Group')['Churn'].mean().mul(100).round(2))

# -----------------------------------------------------------------
# 4. FIGURE 1: Churn balance + correlation heatmap
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.countplot(data=df, x='Churn', hue='Churn', palette=['#4C72B0', '#DD8452'],
              legend=False, ax=axes[0])
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Retained (0)', 'Churned (1)'])
axes[0].set_title(f'Churn Distribution (Churn rate = {churn_rate:.1f}%)')
axes[0].set_ylabel('Number of Customers')
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height()):,}', (p.get_x()+p.get_width()/2, p.get_height()),
                      ha='center', va='bottom')

num_cols = ['Age','Tenure','Usage Frequency','Support Calls','Payment Delay',
            'Total Spend','Last Interaction','Churn']
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1], cbar=True)
axes[1].set_title('Correlation Matrix (Numeric Features)')

plt.tight_layout()
plt.savefig('figures/fig1_churn_balance_correlation.png', bbox_inches='tight')
plt.show()
plt.close()

# -----------------------------------------------------------------
# 5. FIGURE 2: Distribution of key numeric features split by churn
# -----------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
features = ['Age','Tenure','Support Calls','Payment Delay','Total Spend','Usage Frequency']
for ax, feat in zip(axes.flat, features):
    sns.kdeplot(data=df, x=feat, hue='Churn', fill=True, common_norm=False,
                palette={0:'#4C72B0', 1:'#DD8452'}, alpha=0.4, ax=ax, legend=(feat=='Age'))
    ax.set_title(f'{feat} by Churn Status')
if axes.flat[0].get_legend():
    axes.flat[0].get_legend().set_title('Churn')
plt.tight_layout()
plt.savefig('figures/fig2_distributions_by_churn.png', bbox_inches='tight')
plt.show()
plt.close()

# -----------------------------------------------------------------
# 6. FIGURE 3: Churn rate by categorical segments
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
cats = ['Gender','Subscription Type','Contract Length']
for ax, cat in zip(axes, cats):
    rate = df.groupby(cat)['Churn'].mean().mul(100).sort_values(ascending=False)
    sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette='viridis', legend=False, ax=ax)
    ax.set_title(f'Churn Rate (%) by {cat}')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, 100)
    for i, v in enumerate(rate.values):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.savefig('figures/fig3_churn_by_segment.png', bbox_inches='tight')
plt.show()
plt.close()

# -----------------------------------------------------------------
# 7. FIGURE 4: Boxplots of strongest churn drivers
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, feat in zip(axes, ['Support Calls','Payment Delay','Total Spend']):
    sns.boxplot(data=df, x='Churn', y=feat, hue='Churn', palette=['#4C72B0','#DD8452'], legend=False, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Retained','Churned'])
    ax.set_title(f'{feat} vs Churn')
plt.tight_layout()
plt.savefig('figures/fig4_boxplots_key_drivers.png', bbox_inches='tight')
plt.show()
plt.close()

print("\nAll Task 1 figures saved successfully.")
