# 🤖 AI Workflow Assignment – End-to-End ML Pipeline

Welcome to the official repository for the **AI Workflow Assignment** — a collaborative project built as part of the PLP Academy program.

This project walks through the **full AI development lifecycle**, covering everything from problem scoping to data preprocessing, model training, evaluation, deployment, and ethical reflection.

---

### 📎 Medium Article

📖 Full write-up with insights and breakdowns:  
➡️ [Read on Medium →](https://medium.com/@codegnerdev/a-hands-on-walkthrough-of-a-complete-ai-pipeline-from-data-preprocessing-to-model-deployment-c0b28d3d6713)

---

## 👨‍💻 Contributors

- **Fred Kibutu** – Model implementation, evaluation, deployment, and API development
- **Dadius Ainda** – Problem definition, ethical analysis, and hospital readmission case study

---

## 📌 Project Objectives

- Predict student academic performance (binary classification based on final grade)
- Simulate real-world AI development with proper documentation, versioning, and deployment
- Explore a secondary case study: predicting hospital readmission
- Practice ethical AI thinking (bias, fairness, and compliance)

---

## 🧠 Contributors

- **Fred Kibutu (@KibutuJr)** – Model implementation, evaluation, FastAPI deployment
- **[Partner Name]** – AI problem framing, healthcare case study, ethics and strategy

---

## 🧱 Project Structure

```
AI-Workflow/
├── data/
│ ├── student-por.csv # Raw dataset from UCI
│ └── clean.csv # Preprocessed dataset
├── models/
│ ├── readmit_model.pkl # Trained RandomForestClassifier
│ └── features.txt # Feature order for inference
├── src/
│ ├── preprocess.py # Data cleaning and encoding
│ ├── train.py # Model training script
│ ├── evaluate.py # Confusion matrix, precision, recall
│ └── deploy_api.py # FastAPI model API
├── diagram.drawio # AI Workflow visual diagram
├── requirements.txt # Python dependencies
├── .gitignore # Ignore venv, cache files, etc.
├── README.md # This file
└── AI Problem Analysis Report.pdf
└──report.md # Full write-up

```

---

## 🔧 Setup Instructions

### 1. Clone the repository

git clone https://github.com/KibutuJr/AI-Workflow.git
cd AI-Workflow

### 2. Set up virtual environment

python -m venv .venv
source .venv/Scripts/activate # Windows

## OR

source .venv/bin/activate # macOS/Linux

### 3. Install required packages

pip install -r requirements.txt

## 🧪 Run the Pipeline

### Step 1: Preprocess the dataset

python src/preprocess.py

### Step 2: Train the model

python src/train.py

### Step 3: Evaluate the model

python src/evaluate.py

---

## 🌐 Deploy the API

uvicorn src.deploy_api:app --reload

Go to: http://127.0.0.1:8000/docs
Use the interactive Swagger UI to test /predict.

### Example request:

{
"features": [0.25, 0.75, 0.5, ..., 1, 0] // total of 41 features
}

---

## 📊 Model Performance

### Confusion Matrix:

```
[[44  9]
 [ 1 44]]
```

-- Precision: 0.83

-- Recall: 0.98

-- Model: Random Forest with 100 trees, max depth = 5

---

### 📉 AI Development Workflow Diagram

-- Located in: `diagram.drawio/diagram.drawio`  
-- Exported as: `diagram.drawio/Diagram Drawio.png`

This diagram outlines the complete AI workflow followed in this project:

![AI Workflow Diagram](diagram.drawio/Diagram%20Drawio.png)

**Stages covered:**
1. Problem Definition  
2. Data Collection  
3. Preprocessing  
4. Model Training  
5. Evaluation  
6. Deployment  
7. Monitoring & Feedback  
8. Ethics & Bias Review


---

### 📚 References

- Pavan, S. (2017). IBM HR Analytics Employee Attrition Dataset. Kaggle. https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

- Chicco, D., & Jurman, G. (2021). Electronic health records and ML. Journal of Biomedical Informatics. https://doi.org/10.1016/j.jbi.2021.103701

- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Oversampling Technique. JAIR. http://www.jair.org/media/953/live-953-2037-jair.pdf

- U.S. Department of Health & Human Services. (1996). HIPAA Privacy Rule. https://www.hhs.gov/hipaa/for-professionals/privacy/index.html

- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR. http://jmlr.org/papers/v12/pedregosa11a.html

---

### 💡 Reflection

This project helped us practice every phase of building an AI system—not just coding, but also ethics, deployment, and stakeholder thinking. It challenged us to ensure data was clean, models were fair, and systems were deployable.

---

### 📄 License

This project is part of the PLP Academy curriculum and is released for non-commercial, educational use only.

### Thanks for exploring!

---
