# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using three classification models — Logistic Regression, Random Forest, and XGBoost — with SMOTE oversampling to handle severe class imbalance.

---

## 📊 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by ULB Machine Learning Group

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.17%) |
| Legitimate transactions | 284,315 (99.83%) |
| Features | 30 (V1–V28 are PCA-transformed, plus Time and Amount) |
| Missing values | None |

> ⚠️ The dataset file `creditcard.csv` is not included in this repository (143 MB). Download it from Kaggle and place it in the project root before running.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/HarshittSinghrowa/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
```

### 3. Download the dataset
- Go to [Kaggle dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Download and unzip `creditcard.csv`
- Place it in the project root folder

### 4. Run the model
```bash
python fraud_detection.py
```

---

## 🧠 How It Works

### Pipeline Overview

```
Raw Data (creditcard.csv)
        │
        ▼
  Preprocessing
  • Scale Amount & Time
  • V1–V28 already PCA-transformed
        │
        ▼
  Train / Test Split (80% / 20%, stratified)
        │
        ▼
  SMOTE Oversampling (training set only)
  • Synthetic minority oversampling
  • Balances the 0.17% fraud class
        │
        ▼
  ┌─────────────────────────────────┐
  │  Three Models Trained           │
  │  • Logistic Regression          │
  │  • Random Forest (100 trees)    │
  │  • XGBoost (100 estimators)     │
  └─────────────────────────────────┘
        │
        ▼
  Evaluation
  • ROC-AUC, PR-AUC
  • Precision, Recall, F1
  • Confusion Matrix
  • Feature Importance
```

### Why SMOTE?
The dataset is severely imbalanced — only 0.17% of transactions are fraudulent. Without correction, models learn to predict "legitimate" for everything and achieve 99.83% accuracy while catching zero fraud. SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic fraud samples during training to balance the classes.

### Why PR-AUC over Accuracy?
In imbalanced datasets, accuracy is misleading. A model predicting "legitimate" for every transaction scores 99.83% accuracy but is completely useless. Precision-Recall AUC directly measures the trade-off between catching fraud (recall) and avoiding false alarms (precision) — a far more honest metric.

---

## 📈 Results

### Model Comparison

| Model | ROC-AUC | PR-AUC | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.9698 | 0.7249 | 0.06 | 0.92 | 0.11 |
| Random Forest | 0.9688 | **0.8678** | **0.82** | 0.82 | **0.82** |
| XGBoost | **0.9781** | 0.8397 | 0.24 | **0.89** | 0.38 |

### Interpretation

**Random Forest** is the most production-ready model:
- 82% precision — when it flags fraud, it's correct 82% of the time
- 82% recall — catches 82% of all actual fraud cases
- Best PR-AUC (0.87), meaning the best balance across all thresholds

**XGBoost** has the highest ROC-AUC but low precision (24%) — it catches more fraud but generates more false alarms, meaning more legitimate transactions get wrongly blocked.

**Logistic Regression** catches the most fraud (92% recall) but has extremely low precision (6%) — for every real fraud flagged, ~16 legitimate transactions are also blocked.

### Choosing a Model Based on Business Need

| Priority | Recommended Model |
|---|---|
| Minimise false alarms (customer experience) | Random Forest |
| Catch as much fraud as possible | Logistic Regression |
| Best overall discrimination | XGBoost |

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── fraud_detection.py       # Main ML pipeline script
├── README.md                # This file
├── .gitignore               # Excludes creditcard.csv and outputs
│
├── model_evaluation.png     # Confusion matrices + PR curves (generated)
├── roc_curves.png           # ROC curves for all models (generated)
└── feature_importance.png   # Top 15 features (Random Forest) (generated)
```

> Generated `.png` files appear after running the script.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models, metrics, preprocessing |
| `imbalanced-learn` | SMOTE oversampling |
| `xgboost` | XGBoost classifier |
| `matplotlib` | Plotting |
| `seaborn` | Heatmaps (confusion matrices) |

**Python version:** 3.8+

---

## 📉 Output Files

After running `fraud_detection.py`, three plots are saved:

| File | Contents |
|---|---|
| `model_evaluation.png` | Confusion matrix + Precision-Recall curve for each model |
| `roc_curves.png` | ROC curves for all three models overlaid |
| `feature_importance.png` | Top 15 most important features from Random Forest |

---

## 🔍 Key Concepts

**ROC-AUC** — Area under the Receiver Operating Characteristic curve. Measures how well a model separates classes across all thresholds. 1.0 = perfect, 0.5 = random.

**PR-AUC** — Area under the Precision-Recall curve. More informative than ROC-AUC when classes are imbalanced. Directly captures the trade-off between catching fraud and avoiding false alarms.

**SMOTE** — Synthetic Minority Over-sampling Technique. Creates synthetic examples of the minority class (fraud) by interpolating between existing fraud samples, rather than simply duplicating them.

**PCA Features (V1–V28)** — The original transaction features have been anonymised and transformed using Principal Component Analysis by the dataset authors for confidentiality reasons.

---

## 🛠️ Possible Improvements

- Hyperparameter tuning with `GridSearchCV` or `Optuna`
- Threshold tuning to optimise precision/recall trade-off for a specific business cost
- Ensemble of all three models (stacking/voting)
- Real-time inference API using Flask or FastAPI
- Adding more models: LightGBM, Isolation Forest (anomaly detection approach)

---

## 👤 Author

**Harshit Singh Rowa**
- GitHub: [@HarshittSinghrowa](https://github.com/HarshittSinghrowa)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- Dataset provided by the [Machine Learning Group at ULB](http://mlg.ulb.ac.be) (Université Libre de Bruxelles)
- Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi — original dataset authors
