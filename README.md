# credit-default-prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow)
![Imbalanced](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-brightgreen)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

---

## 🔗 Quick Links (Repo Files)
- 📓 Notebook (end-to-end): [`Project_notebook.ipynb`](./Project_notebook.ipynb)
- 📄 Final report (PDF): [`Credit_Default_Prediction.pdf`](./Credit_Default_Prediction.pdf)
- 🧪 Data prep script: [`data.py`](./data.py)
- 🖥️ Streamlit app: [`streamlit_app/`](./streamlit_app/)

---

## 🧭 Project Overview
Credit risk teams need early signals of potential delinquency to reduce losses and support consistent lending decisions.  
This project builds a **binary classification model** to predict whether a borrower will become **90+ days delinquent within 2 years** (`SeriousDlqin2yrs`). :contentReference[oaicite:1]{index=1}

---

## 🎯 Project Goals
- Predict future delinquency to strengthen risk monitoring  
- Reduce financial losses by identifying high-risk borrowers early  
- Provide consistent and responsible lending decisions  
- Deliver actionable insights into key risk drivers  
- Compare multiple ML models and select the best-performing approach :contentReference[oaicite:2]{index=2}

---

## 🗂 Dataset
- **Source:** Kaggle “Give Me Some Credit”
- **Size:** ~150,000 records
- **Task:** Binary classification (`SeriousDlqin2yrs`)
- **Features (examples):**
  - Revolving credit utilization, debt ratio, monthly income
  - Late payment counts (30–59 / 60–89 / 90+ days)
  - Open credit lines, real estate loans, dependents :contentReference[oaicite:3]{index=3}

---

## 🧹 Data Preprocessing
### 1) Missing Values
- Missingness concentrated in **MonthlyIncome** and **NumberOfDependents**
- Notebook includes imputation steps to preserve dataset size and avoid biased deletion.

### 2) Outliers
- Outliers reviewed via boxplots for key numeric fields (e.g., income, debt ratio, utilization, dependents)
- Cleaning decisions applied before modeling to stabilize training.

### 3) Class Imbalance
- Target distribution is imbalanced (majority non-delinquency).
- Training pipeline uses **SMOTE** to improve minority-class learning and evaluation stability.

---

## 🔎 Exploratory Data Analysis (EDA) Highlights
- **Target imbalance** visualized (delinquency is the minority class)
- Distributions explored for:
  - Age
  - Monthly income (including tail behavior)
  - Late payment behavior (30–59 / 60–89 / 90+ days)
- Correlation heatmap used to inspect feature relationships and redundancy. :contentReference[oaicite:4]{index=4}

---

## 🏗 Modeling Framework
We evaluated multiple models using a consistent pipeline approach.

### Models Compared
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- MLP Classifier :contentReference[oaicite:5]{index=5}

### Pipeline Design (Notebook)
- **Scaling + SMOTE** applied for models sensitive to feature scale (e.g., LR/MLP)
- **SMOTE-only** used for tree models
- Evaluation centered on **ROC-AUC** (robust for class imbalance)

---

## 🧪 Hyperparameter Tuning
- **GridSearchCV** used for model selection and tuning
- Example tuned parameters (vary by model):
  - Logistic Regression: `C`, `class_weight`
  - Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `class_weight`
  - Gradient Boosting: `n_estimators`, `learning_rate`, `max_depth`
  - MLP: `alpha`, `learning_rate_init`, `hidden_layer_sizes` :contentReference[oaicite:6]{index=6}

---

## 📈 Key Results
**Best-performing model: MLP Classifier**

| Model | Best CV AUC | Test AUC |
|---|---:|---:|
| MLP Classifier | 0.851 | 0.840 |
| Random Forest | 0.829 | 0.822 |
| Gradient Boosting | 0.827 | 0.816 |
| Logistic Regression | 0.800 | 0.795 | :contentReference[oaicite:7]{index=7}

### 🔍 Interpretability (Feature Importance / Reduced Feature Set)
The project also explores feature importance and a reduced-feature evaluation (see report + notebook) to balance performance and explainability.

---

## 🖥️ Streamlit App (Local Demo)
This repository includes a Streamlit demo app under `streamlit_app/`.

> Note: GitHub does not run Streamlit apps. Run it locally.

### Run locally
```bash
pip install streamlit
streamlit run streamlit_app/app.py
