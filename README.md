# 🏥 ChroniCare: Advanced Diagnostics & Clinical Decision Support System

[![Streamlit App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%23154f82?style=for-the-badge)](https://xgboost.readthedocs.io/)

ChroniCare is an intelligent, high-fidelity medical diagnostics and lifestyle intervention platform. It leverages machine learning classifiers, unsupervised cohort segmentation, and reinforcement learning agents to assist clinicians and patients in screening and managing four major chronic conditions: **Cardiovascular Disease**, **Diabetes**, **Chronic Kidney Disease (CKD)**, and **Liver Disorders**.

Developed with a premium, responsive glassmorphic UI, the application provides real-time dual-model risk validation, interactive cohort exploration, treatment pathway optimization, and auto-generated clinician-ready PDF medical reports.

---

## 🌟 Key Features

### 1. Multi-Disease Clinical Screening
Run patient diagnostic screening using dual-model validation (primary optimized classifier vs. baseline classifier) to cross-verify risk scores.
*   ❤️ **Cardiovascular Risk:** Predicts coronary artery disease based on lipid panels, electrocardiogram (ECG) features, and hemodynamic parameters using an **RBF SVM** and **Decision Tree**.
*   🩸 **Diabetes Detection:** Screens blood glucose regulation and metabolic markers using a **Random Forest Classifier** and **Logistic Regression**.
*   🫘 **Renal Function (CKD):** Diagnoses early-stage kidney disease through serum chemistry, electrolytes, and urinalysis using **K-Nearest Neighbors (KNN)** and **Gaussian Naive Bayes**.
*   🟡 **Hepatic Health:** Assesses liver enzyme elevation, bilirubin clearance, and protein synthesis using **XGBoost** and **Gradient Boosting**.

### 2. Unsupervised Patient Cohort Analytics
Visualizes where a patient fits in the overall clinical landscape rather than simplifying them to binary diagnostics:
*   **3D PCA Mapping:** Project high-dimensional (8D) patient records into a 3D visual workspace to inspect metabolic density.
*   **K-Means Segmentation:** Automatically groups patients into three distinct clusters: *Low Risk (Maintenance)*, *Moderate Risk (Elevated)*, and *High Risk (Severe)*.
*   **DBSCAN Anomaly Detection:** Flags atypical demographic or biological outliers that don't fit standard disease profiles.

### 3. Reinforcement Learning Lifestyle Advisor
A simulator featuring a **Q-Learning agent** that maps out optimal, step-by-step treatment and recovery pathways:
*   **Discretized Biological States (27):** Encoded from combinations of blood glucose levels, BMI, and blood pressure.
*   **Intervention Actions:** Simulates the physiological impact of *Carb Reduction*, *Daily Aerobic Exercise*, *Rigorous Vitals Monitoring*, and *Clinical Consultation*.
*   **Reward Maximization:** The agent uses Bellman equation policy updates to guide patients toward metabolic normalization.

### 4. Interactive Data Explorer & Algorithm Comparison
*   **Data Explorer:** Offers real-time histograms, scatterplots, correlation matrices, and distribution metrics for each of the underlying datasets.
*   **Algorithm Comparison:** Compares test accuracy, precision, recall, F1-scores, and ROC-AUC curves of all training models side-by-side.

### 5. Automated PDF Report Generation
Generate and download high-quality, professional PDF reports on-demand for both individual patients and platform summaries using **ReportLab**:
*   Detailed risk assessments and model probability scores.
*   Highlighted out-of-range vitals with detailed clinical context.
*   Evidence-based recovery plans, recommended medication classes, and follow-up schedules.

---

## 📂 Project Structure

```directory
ChroniCare/
├── .streamlit/
│   └── config.toml          # Custom theme configuration (Dark/Light setup)
├── assets/
│   ├── banner.png           # UI landing page hero image
│   ├── logo.png             # ChroniCare branding identity
│   ├── style.css            # Custom CSS styles (glassmorphism, animations)
│   └── style_loader.py      # Python helper applying responsive CSS styles
├── data/                    # Clinical screening datasets (CSV)
│   ├── diabetes.csv         # PIMA Indian Diabetes Dataset
│   ├── heart.csv            # Cleveland Heart Disease Dataset
│   ├── kidney.csv           # UCI Machine Learning Repository CKD Dataset
│   └── liver.csv            # Indian Liver Patient Dataset (ILPD)
├── models/                  # Serialized ML models & metrics (PKL/NPY)
│   ├── dbscan_diabetes.pkl  # DBSCAN / K-Means serialized coordinates
│   ├── diabetes_rf.pkl      # Random Forest model for Diabetes
│   ├── diabetes_lr.pkl      # Logistic Regression model for Diabetes
│   ├── heart_svm.pkl        # Support Vector Machine for Heart Disease
│   ├── heart_dt.pkl         # Decision Tree model for Heart Disease
│   ├── kidney_knn.pkl       # KNN model for Kidney Disease
│   ├── kidney_nb.pkl        # Naive Bayes model for Kidney Disease
│   ├── liver_xgb.pkl        # XGBoost model for Liver Disease
│   ├── liver_gb.pkl         # Gradient Boosting model for Liver Disease
│   ├── kmeans_diabetes.pkl  # K-Means clustering model for Diabetes
│   ├── pca_diabetes.pkl     # Principal Component Analysis coordinates
│   ├── q_table.npy          # Trained Reinforcement Learning Q-Table
│   └── model_metrics.pkl    # Serialized accuracy/AUC stats for comparisons
├── notebooks/               # Development Jupyter Notebooks
│   ├── supervised_training.ipynb
│   ├── unsupervised_training.ipynb
│   └── rl_training.ipynb
├── pages/                   # Application Multi-page Router files
│   ├── 1_home.py            # Dashboard main landing layout
│   ├── 2_diabetes.py        # Diabetes screening & batch diagnostics
│   ├── 3_heart.py           # Heart disease screening interface
│   ├── 4_kidney.py          # Chronic kidney disease screening interface
│   ├── 5_liver.py           # Liver disease screening interface
│   ├── 6_analytics.py       # Unsupervised cohort & 3D PCA page
│   ├── 7_rl_advisor.py      # Q-learning lifestyle treatment advisor simulator
│   ├── 8_comparison.py      # Model benchmarking curves comparison
│   ├── 9_data_explorer.py   # Dataset correlation & distribution viewer
│   ├── 10_about_report.py   # System metadata overview page
│   └── 11_patient_report_portal.py  # Patient history logs & report search engine
├── app.py                   # Main entry point & routing configuration
├── data_generator.py        # Synthetic patient history generator
├── generate_report.py       # Core PDF compilation engine for platform analytics
├── patient_history.csv      # Log database storing diagnostic histories
├── patient_report_generator.py # Patient-specific PDF report engine
├── requirements.txt         # Core project libraries & dependencies
├── styles.py                # System-wide style injection wrapper
├── train.py                 # Pipeline training & model serialization script
└── utils.py                 # Common helper utilities & patient history log handlers
```

---

## ⚙️ Model Training & Deep Dive

### Supervised Classification Details
The models are trained using a $80/20$ train-test split, scaled using `StandardScaler` to normalize features:
*   **Diabetes:** `Random Forest` (n_estimators=100, max_depth=8) vs. `Logistic Regression`.
*   **Heart Disease:** `SVM` (RBF kernel, probability enabled) vs. `Decision Tree` (max_depth=5).
*   **Kidney Disease:** `K-Nearest Neighbors` (n_neighbors=5) vs. `Gaussian Naive Bayes`.
*   **Liver Disease:** `XGBoost Classifier` (max_depth=4, learning_rate=0.1) vs. `Gradient Boosting Classifier` (max_depth=4).

### Reinforcement Learning MDP Setup
*   **State Space ($S=27$):** Decoded from Glucose $[0,1,2] \times$ BMI $[0,1,2] \times$ Blood Pressure $[0,1,2]$.
*   **Action Space ($A=4$):** Carb Reduction ($0$), Incorporate Aerobic Exercise ($1$), Monitor Vitals ($2$), Clinical Consultation ($3$).
*   **Transition Reward Policy:**
    *   Actions that shift states closer to optimal health (e.g. Carb Reduction when glucose is high) yield $+15$ to $+20$ reward.
    *   Remaining in bad metabolic states applies heavy health penalties: $Reward \leftarrow Reward - (Glucose \times 6 + BMI \times 4 + BP \times 5)$.
*   **Update Rule:** Bellman Optimality Equation:
    $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
    *   *Parameters:* Learning rate $\alpha=0.1$, Discount factor $\gamma=0.9$, Exploration rate $\epsilon=0.3$. Trained over 8,000 learning episodes.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed.

### 2. Install Dependencies
Clone the repository, navigate to the project folder, and run:
```bash
pip install -r requirements.txt
```

### 3. Train and Serialize Models
Train the supervised classifiers, clustering algorithms, and the RL advisor agent:
```bash
python train.py
```
This script will output the trained model binary files inside the `models/` directory.

### 4. Generate the Platform Technical PDF Report
Compile the technical overview of the system:
```bash
python generate_report.py
```
This saves `report.pdf` in the root folder.

### 5. Launch the Web Application
Run the Streamlit server:
```bash
streamlit run app.py
```
Access the dashboard on your browser at `http://localhost:8501`.

---

## 🛠️ Technology Stack
*   **Interface:** [Streamlit](https://streamlit.io)
*   **Data Analysis & Modeling:** Pandas, NumPy, Scikit-Learn, XGBoost
*   **Visualization:** Plotly Express, Matplotlib, Seaborn
*   **Reporting:** ReportLab (automated clinical PDF compilation)
*   **Styling:** Custom CSS Injection (Glassmorphism card effects, animated radial pulses, dynamic sidebar routing tags)

---

## 📋 Disclaimer
*ChroniCare is a clinical decision support prototype developed for training, research, and informational demonstrations. It is not an FDA-approved diagnostic tool and does not replace medical advice, diagnoses, or treatments from a licensed healthcare professional.*
