# 📄 AI Development Workflow Assignment

**Project Title:** Predicting Student Academic Performance  
**Author:** Fred Kibutu & Dadius Ainda
**Date:** June 28, 2025

---

## 1. 🧠 Problem Definition

**Objective:**  
Design and deploy a machine learning system that predicts whether a student is likely to score above the median in their final grade (G3) based on demographic, academic, and behavioral data.

**Goals:**

- Identify students at risk of underperforming.
- Assist teachers in targeting support resources.
- Enable data-driven academic planning.

**Stakeholders:**

- **Teachers & School Administrators** — to intervene early.
- **Students & Parents** — to receive personalized support.

**Success KPI:**

- **F1 Score** (harmonic mean of precision and recall) > 0.85

---

## 2. 🗃️ Data Sources

We use the **Student Performance Data Set** from the **UCI Machine Learning Repository**, which contains:

- Demographic info (e.g., age, sex, address)
- Academic background (grades G1, G2, G3)
- Social behavior (freetime, goout, absences)
- Family and school context

**File used:** `student-por.csv`

---

## 3. 🧼 Preprocessing Steps

Performed in `src/preprocess.py`:

### ✅ Key Preprocessing Actions:

- **Missing Values:** Checked and confirmed none in original data.
- **Label Encoding / One-Hot Encoding:**
  - Binary categorical variables: label encoded
  - Multiclass categorical variables: one-hot encoded
- **Feature Engineering:**
  - Converted G3 (final grade) to binary target: `1` if above median, else `0`
- **Scaling:**
  - Applied Min-Max Scaling to numeric features
- **Output:** `data/clean.csv`

---

## 4. 🧠 Model Development

### ✅ Model Used: `RandomForestClassifier`

**Justification:**

- Handles both numerical and categorical features well.
- Naturally robust to overfitting with parameter tuning.
- Easy to interpret feature importance.

### 🔧 Hyperparameters Tuned:

- `n_estimators = 100` → number of trees
- `max_depth = 5` → limits overfitting

### 📊 Dataset Splitting:

- **Train:** 70%
- **Validation:** 15%
- **Test:** 15%

### 📁 Model Saved To:

- `models/readmit_model.pkl`

---

## 5. 📊 Evaluation Summary

Performed in `src/evaluate.py`:

### 🔁 Confusion Matrix:

` [[44 9]
[ 1 44]]`

- **Precision:** 0.83
- **Recall:** 0.98

> The model is excellent at catching high-performing students (high recall), while maintaining reasonable precision.

---

## 6. 🚀 Deployment Strategy

Deployment implemented using **FastAPI** in `src/deploy_api.py`:

### ✅ Steps:

- Load trained model from `.pkl`
- Accept `features` list via `/predict` endpoint
- Return readmission probability (as JSON)
- Hosted locally via Uvicorn: `http://127.0.0.1:8000/docs`

### 🔧 Input:

- Must match 41 features (order from training)
- Example API call via Swagger or curl

### 📦 Requirements:

- `uvicorn`, `fastapi`, `scikit-learn`, `pandas`, etc. listed in `requirements.txt`

---

## 7. ⚖️ Ethics & Bias

### 🔍 Potential Bias:

- Socioeconomic, gender, or family structure biases may be reflected in historical data.
- Overrepresentation of some groups could lead to unfair predictions.

### ✅ Mitigation Strategies:

- Use **balanced datasets**
- Continuously monitor model fairness post-deployment
- Avoid over-relying on features like parental job, school type, or family size

---

## 8. 🧭 Workflow Diagram

See `diagram.drawio`

**Workflow Stages:**

1. 🧠 Problem Definition
2. 🗃️ Data Collection
3. 🧼 Data Preprocessing
4. 🧠 Model Training
5. 📊 Evaluation
6. 🚀 Deployment (API)
7. 🔁 Monitoring & Feedback

---

## 9. 🪞 Reflection

### 📌 Most Challenging Part:

Ensuring all 41 input features matched exactly between preprocessing and prediction was tricky. A mismatch caused multiple errors in deployment and required careful inspection.

### 💡 What I’d Improve:

With more time, I’d:

- Add automated data validation in the API.
- Deploy via Docker and test with live web clients.
- Use model explainability tools like SHAP for transparency.

---

## ✅ Conclusion

This assignment provided hands-on practice across the **entire AI lifecycle**: from problem framing to real-world deployment. The project is fully reproducible and documented via GitHub.
