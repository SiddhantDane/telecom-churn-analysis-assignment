import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# NOTE: no matplotlib.use('Agg') here on purpose — see task1_eda.py for why.
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix, classification_report,
                              silhouette_score)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 110
os.makedirs('figures', exist_ok=True)
t0 = time.time()

# -----------------------------------------------------------------
# 1. LOAD CLEANED DATA (output of task1_eda.py) & FEATURE ENGINEERING
# -----------------------------------------------------------------
df = pd.read_csv('churn_cleaned.csv')

# CustomerID is a unique row identifier with no real-world causal link to churn
# (EDA showed an artificial -0.84 correlation purely from row ordering) -> exclude
model_df = df.drop(columns=['CustomerID']).copy()
model_df['Gender'] = model_df['Gender'].map({'Female': 1, 'Male': 0})
model_df = pd.get_dummies(model_df, columns=['Subscription Type', 'Contract Length'], drop_first=True)

X = model_df.drop(columns=['Churn'])
y = model_df['Churn']
feature_names = X.columns.tolist()
print("Features used:", feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[{time.time()-t0:.1f}s] data prepared. Train={X_train.shape}, Test={X_test.shape}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# -----------------------------------------------------------------
# MODEL 1: LOGISTIC REGRESSION (tuned on full training data - cheap model)
# -----------------------------------------------------------------
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42),
                        param_grid={'C': [0.01, 0.1, 1, 10]}, cv=3, scoring='f1', n_jobs=1)
lr_grid.fit(X_train_scaled, y_train)
best_lr = lr_grid.best_estimator_
print(f"[{time.time()-t0:.1f}s] LR tuned. Best params={lr_grid.best_params_}")

lr_pred = best_lr.predict(X_test_scaled)
lr_proba = best_lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = dict(
    accuracy=accuracy_score(y_test, lr_pred), precision=precision_score(y_test, lr_pred),
    recall=recall_score(y_test, lr_pred), f1=f1_score(y_test, lr_pred),
    roc_auc=roc_auc_score(y_test, lr_proba), pred=lr_pred, proba=lr_proba)

# -----------------------------------------------------------------
# MODEL 2: RANDOM FOREST
# Tuned on a 40k-row subsample first (keeps grid search fast on machines
# with few CPU cores), then the best params are refit on the FULL training
# set, which is standard practice for large datasets.
# -----------------------------------------------------------------
X_sub = X_train.sample(40000, random_state=42)
y_sub = y_train.loc[X_sub.index]

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=1),
                        param_grid={'n_estimators': [50, 100], 'max_depth': [10, 15],
                                    'min_samples_leaf': [1, 5]},
                        cv=3, scoring='f1', n_jobs=1)
rf_grid.fit(X_sub, y_sub)
print(f"[{time.time()-t0:.1f}s] RF tuned on 40k subsample. Best params={rf_grid.best_params_}")

best_rf = RandomForestClassifier(**rf_grid.best_params_, random_state=42, n_jobs=1)
best_rf.fit(X_train, y_train)
print(f"[{time.time()-t0:.1f}s] RF refit on full training set ({len(X_train)} rows)")

rf_pred = best_rf.predict(X_test)
rf_proba = best_rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = dict(
    accuracy=accuracy_score(y_test, rf_pred), precision=precision_score(y_test, rf_pred),
    recall=recall_score(y_test, rf_pred), f1=f1_score(y_test, rf_pred),
    roc_auc=roc_auc_score(y_test, rf_proba), pred=rf_pred, proba=rf_proba)

# -----------------------------------------------------------------
# RESULTS TABLE
# -----------------------------------------------------------------
results_table = pd.DataFrame({n: {k: v for k, v in m.items() if k in ['accuracy','precision','recall','f1','roc_auc']}
                               for n, m in results.items()}).T.round(4)
print("\n--- MODEL COMPARISON ---\n", results_table)
results_table.to_csv('model_comparison.csv')

print("\n--- Classification Report: Logistic Regression ---")
print(classification_report(y_test, lr_pred, target_names=['Retained', 'Churned']))
print("--- Classification Report: Random Forest ---")
print(classification_report(y_test, rf_pred, target_names=['Retained', 'Churned']))

# -----------------------------------------------------------------
# FIGURES: confusion matrices, ROC, feature importance
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, pred) in zip(axes, [('Logistic Regression', lr_pred), ('Random Forest', rf_pred)]):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
    ax.set_title(f'Confusion Matrix: {name}'); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout(); plt.savefig('figures/fig5_confusion_matrices.png', bbox_inches='tight'); plt.show(); plt.close()

plt.figure(figsize=(7, 6))
for name, m in results.items():
    fpr, tpr, _ = roc_curve(y_test, m['proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={m['roc_auc']:.3f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Guess')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curves')
plt.legend(); plt.tight_layout(); plt.savefig('figures/fig6_roc_curves.png', bbox_inches='tight'); plt.show(); plt.close()

importances = pd.Series(best_rf.feature_importances_, index=feature_names).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='mako', legend=False)
plt.title('Random Forest: Feature Importance'); plt.xlabel('Importance'); plt.ylabel('Feature')
plt.tight_layout(); plt.savefig('figures/fig7_feature_importance.png', bbox_inches='tight'); plt.show(); plt.close()
print("\nFeature importance:\n", importances.round(4))

# -----------------------------------------------------------------
# K-MEANS CLUSTERING (3rd model -> customer segmentation for Tableau/Task 4)
# -----------------------------------------------------------------
cluster_features = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay',
                     'Total Spend', 'Last Interaction']
scaler_c = StandardScaler()
X_cluster_full = scaler_c.fit_transform(df[cluster_features])

# Elbow/silhouette on a 30k subsample to choose k quickly
sub_idx = df.sample(30000, random_state=42).index
X_cluster_sub = X_cluster_full[sub_idx]
inertias, sil_scores = [], []
K_range = range(2, 7)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster_sub)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_cluster_sub, km.labels_))
print(f"[{time.time()-t0:.1f}s] KMeans k-selection done. Silhouette by k:",
      dict(zip(K_range, np.round(sil_scores, 4))))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(K_range), inertias, marker='o')
axes[0].set_title('Elbow Method'); axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[1].plot(list(K_range), sil_scores, marker='o', color='darkorange')
axes[1].set_title('Silhouette Score'); axes[1].set_xlabel('k')
plt.tight_layout(); plt.savefig('figures/fig8_kmeans_selection.png', bbox_inches='tight'); plt.show(); plt.close()

final_k = 4
kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10).fit(X_cluster_full)
df['Cluster'] = kmeans_final.labels_
print(f"[{time.time()-t0:.1f}s] final KMeans (k=4) fit on full data")

cluster_profile = df.groupby('Cluster')[cluster_features + ['Churn']].mean().round(2)
cluster_profile['Customers'] = df['Cluster'].value_counts().sort_index()
print("\n--- Cluster Profiles ---\n", cluster_profile)
cluster_profile.to_csv('cluster_profile.csv')

# PCA projection purely for 2D visualisation of the 6-dimensional clusters
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_cluster_full)
df['PC1'], df['PC2'] = coords[:, 0], coords[:, 1]

plt.figure(figsize=(8, 6))
sample = df.sample(15000, random_state=42)
sns.scatterplot(data=sample, x='PC1', y='PC2', hue='Cluster', palette='Set2', alpha=0.5, s=18)
plt.title(f'Customer Segments (K-Means, k=4) — PCA Projection\n'
          f'({pca.explained_variance_ratio_.sum()*100:.1f}% variance explained)')
plt.tight_layout(); plt.savefig('figures/fig9_kmeans_clusters.png', bbox_inches='tight'); plt.show(); plt.close()

df.to_csv('churn_for_tableau.csv', index=False)

print(f"\n[{time.time()-t0:.1f}s] TASK 2 COMPLETE.")
