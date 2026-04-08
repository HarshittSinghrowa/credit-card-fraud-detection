import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, RocCurveDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

# 1. LOAD & EXPLORE DATA
print("=" * 60)
print("  Credit Card Fraud Detection Pipeline")
print("=" * 60)

df = pd.read_csv(r"C:\Users\Harshit\Desktop\CC Fraud Detection\creditcard.csv")

print(f"\nDataset shape: {df.shape}")
print(f"\nClass distribution:")
fraud_counts = df["Class"].value_counts()
print(f"  Legitimate (0): {fraud_counts[0]:,}  ({fraud_counts[0]/len(df)*100:.2f}%)")
print(f"  Fraudulent  (1): {fraud_counts[1]:,}  ({fraud_counts[1]/len(df)*100:.2f}%)")
print(f"\nMissing values: {df.isnull().sum().sum()}")

# 2. PREPROCESSING
# Scale 'Amount' and 'Time' — V1-V28 are already PCA-transformed
scaler = StandardScaler()
df["scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
df["scaled_Time"]   = scaler.fit_transform(df[["Time"]])
df.drop(["Amount", "Time"], axis=1, inplace=True)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

# 3. DEFINE MODELS (with SMOTE for class imbalance)
smote = SMOTE(random_state=42)

models = {
    "Logistic Regression": ImbPipeline([
        ("smote", smote),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": ImbPipeline([
        ("smote", smote),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]),
    "XGBoost": ImbPipeline([
        ("smote", smote),
        ("clf", xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1
        ))
    ]),
}

# 4. TRAIN & EVALUATE
results = {}

print("\n" + "─" * 60)
print("  Training & Evaluating Models")
print("─" * 60)

fig, axes = plt.subplots(len(models), 2, figsize=(14, 5 * len(models)))

for i, (name, pipeline) in enumerate(models.items()):
    print(f"\n {name}...")
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    results[name] = {
        "ROC-AUC": roc_auc,
        "Avg Precision (PR-AUC)": avg_prec,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }

    print(f"  ROC-AUC            : {roc_auc:.4f}")
    print(f"  PR-AUC             : {avg_prec:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    # — Confusion Matrix —
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"],
                ax=axes[i][0])
    axes[i][0].set_title(f"{name}\nConfusion Matrix")
    axes[i][0].set_ylabel("Actual")
    axes[i][0].set_xlabel("Predicted")

    # — Precision-Recall Curve —
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    axes[i][1].plot(recall, precision, label=f"PR-AUC = {avg_prec:.4f}", color="darkorange")
    axes[i][1].set_xlabel("Recall")
    axes[i][1].set_ylabel("Precision")
    axes[i][1].set_title(f"{name}\nPrecision-Recall Curve")
    axes[i][1].legend()
    axes[i][1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
print("\n Saved evaluation plots → model_evaluation.png")

# 5. ROC CURVES — ALL MODELS TOGETHER
fig2, ax = plt.subplots(figsize=(8, 6))
for name, res in results.items():
    RocCurveDisplay.from_predictions(
        y_test, res["y_proba"], name=f"{name} (AUC={res['ROC-AUC']:.4f})", ax=ax
    )
ax.set_title("ROC Curves — All Models")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
print(" Saved ROC curves → roc_curves.png")

# 6. FEATURE IMPORTANCE (Random Forest)
rf_model = models["Random Forest"].named_steps["clf"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top15 = importances.nlargest(15)

fig3, ax = plt.subplots(figsize=(9, 6))
top15.sort_values().plot(kind="barh", color="steelblue", ax=ax)
ax.set_title("Top 15 Feature Importances (Random Forest)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
print(" Saved feature importance → feature_importance.png")

# 7. SUMMARY TABLE
print("\n" + "=" * 60)
print("  FINAL MODEL COMPARISON")
print("=" * 60)
summary = pd.DataFrame({
    name: {k: v for k, v in res.items() if k not in ("y_proba", "y_pred")}
    for name, res in results.items()
}).T
print(summary.to_string())

best = summary["ROC-AUC"].idxmax()
print(f"\nBest model by ROC-AUC: {best} ({summary.loc[best, 'ROC-AUC']:.4f})")
print("\nDone! Check model_evaluation.png, roc_curves.png, feature_importance.png")
