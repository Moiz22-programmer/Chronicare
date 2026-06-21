import os
import json

def create_supervised_notebook(output_path):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ChroniCare: Supervised Model Training Pipeline\n",
                    "This notebook details the training pipelines for the four chronic disease classifiers of ChroniCare:\n",
                    "1. **Diabetes**: Random Forest (Primary) vs. Logistic Regression (Baseline)\n",
                    "2. **Heart Disease**: Support Vector Machine (Primary) vs. Decision Tree (Baseline)\n",
                    "3. **Kidney Disease (CKD)**: K-Nearest Neighbors (Primary) vs. Gaussian Naive Bayes (Baseline)\n",
                    "4. **Liver Disease**: XGBoost (Primary) vs. Gradient Boosting (Baseline)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc\n",
                    "import pickle\n",
                    "import os"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Diabetes Prediction Model\n",
                    "We train a **Random Forest Classifier** and compare it with a **Logistic Regression** baseline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_diab = pd.read_csv('../data/diabetes.csv')\n",
                    "print(\"Diabetes Dataset Shape:\", df_diab.shape)\n",
                    "X = df_diab.drop(columns=['Outcome'])\n",
                    "y = df_diab['Outcome']\n",
                    "\n",
                    "scaler = StandardScaler()\n",
                    "X_scaled = scaler.fit_transform(X)\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
                    "\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "\n",
                    "rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)\n",
                    "rf.fit(X_train, y_train)\n",
                    "print(\"Random Forest Train Accuracy:\", rf.score(X_train, y_train))\n",
                    "print(\"Random Forest Test Accuracy:\", rf.score(X_test, y_test))\n",
                    "\n",
                    "lr = LogisticRegression(max_iter=1000, random_state=42)\n",
                    "lr.fit(X_train, y_train)\n",
                    "print(\"Logistic Regression Test Accuracy:\", lr.score(X_test, y_test))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Heart Disease Prediction Model\n",
                    "We train a **Support Vector Classifier (SVC)** and compare it with a **Decision Tree Classifier** baseline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_heart = pd.read_csv('../data/heart.csv')\n",
                    "X = df_heart.drop(columns=['target'])\n",
                    "y = df_heart['target']\n",
                    "\n",
                    "scaler_heart = StandardScaler()\n",
                    "X_scaled = scaler_heart.fit_transform(X)\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
                    "\n",
                    "from sklearn.svm import SVC\n",
                    "from sklearn.tree import DecisionTreeClassifier\n",
                    "\n",
                    "svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)\n",
                    "svm.fit(X_train, y_train)\n",
                    "print(\"SVM Test Accuracy:\", svm.score(X_test, y_test))\n",
                    "\n",
                    "dt = DecisionTreeClassifier(max_depth=5, random_state=42)\n",
                    "dt.fit(X_train, y_train)\n",
                    "print(\"Decision Tree Test Accuracy:\", dt.score(X_test, y_test))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Chronic Kidney Disease Prediction Model\n",
                    "We train a **K-Nearest Neighbors Classifier** and compare it with a **Gaussian Naive Bayes** baseline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_kidney = pd.read_csv('../data/kidney.csv')\n",
                    "X = df_kidney.drop(columns=['class'])\n",
                    "y = df_kidney['class']\n",
                    "\n",
                    "scaler_kidney = StandardScaler()\n",
                    "X_scaled = scaler_kidney.fit_transform(X)\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
                    "\n",
                    "from sklearn.neighbors import KNeighborsClassifier\n",
                    "from sklearn.naive_bayes import GaussianNB\n",
                    "\n",
                    "knn = KNeighborsClassifier(n_neighbors=5)\n",
                    "knn.fit(X_train, y_train)\n",
                    "print(\"KNN Test Accuracy:\", knn.score(X_test, y_test))\n",
                    "\n",
                    "nb = GaussianNB()\n",
                    "nb.fit(X_train, y_train)\n",
                    "print(\"Naive Bayes Test Accuracy:\", nb.score(X_test, y_test))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Liver Disease Prediction Model\n",
                    "We train an **XGBoost Classifier** and compare it with a **Gradient Boosting Classifier** baseline."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df_liver = pd.read_csv('../data/liver.csv')\n",
                    "X = df_liver.drop(columns=['Dataset'])\n",
                    "y = df_liver['Dataset']\n",
                    "\n",
                    "scaler_liver = StandardScaler()\n",
                    "X_scaled = scaler_liver.fit_transform(X)\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n",
                    "\n",
                    "from xgboost import XGBClassifier\n",
                    "from sklearn.ensemble import GradientBoostingClassifier\n",
                    "\n",
                    "xgb = XGBClassifier(eval_metric='logloss', max_depth=4, learning_rate=0.1, random_state=42)\n",
                    "xgb.fit(X_train, y_train)\n",
                    "print(\"XGBoost Test Accuracy:\", xgb.score(X_test, y_test))\n",
                    "\n",
                    "gb = GradientBoostingClassifier(max_depth=4, learning_rate=0.1, random_state=42)\n",
                    "gb.fit(X_train, y_train)\n",
                    "print(\"Gradient Boosting Test Accuracy:\", gb.score(X_test, y_test))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)

def create_unsupervised_notebook(output_path):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ChroniCare: Unsupervised Learning & Clustering Analysis\n",
                    "This notebook contains the unsupervised patient clustering and dimensionality reduction models for Diabetes risk group segmentations:\n",
                    "- **K-Means Clustering**: Group segmentation based on patient vitals and indices.\n",
                    "- **Principal Component Analysis (PCA)**: Projecting clinical features down into 2D and 3D visual subspaces.\n",
                    "- **DBSCAN**: Outlier and anomaly detection to flag highly abnormal patient readings."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "from sklearn.preprocessing import StandardScaler\n",
                    "from sklearn.cluster import KMeans, DBSCAN\n",
                    "from sklearn.decomposition import PCA\n",
                    "import seaborn as sns"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Data Scaling"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "df = pd.read_csv('../data/diabetes.csv')\n",
                    "features = df.drop(columns=['Outcome'])\n",
                    "scaler = StandardScaler()\n",
                    "scaled_feats = scaler.fit_transform(features)"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Principal Component Analysis (PCA)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "pca = PCA(n_components=3)\n",
                    "pca_coords = pca.fit_transform(scaled_feats)\n",
                    "print(\"Explained Variance Ratio per Principal Component:\", pca.explained_variance_ratio_)\n",
                    "print(\"Total Explained Variance:\", sum(pca.explained_variance_ratio_))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. K-Means Risk Segmentation"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)\n",
                    "cluster_labels = kmeans.fit_predict(scaled_feats)\n",
                    "features['Cluster'] = cluster_labels\n",
                    "print(features.groupby('Cluster').mean())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. DBSCAN Outlier Detection"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "dbscan = DBSCAN(eps=1.8, min_samples=5)\n",
                    "anomaly_labels = dbscan.fit_predict(scaled_feats)\n",
                    "print(\"Number of outliers detected (Label = -1):\", sum(anomaly_labels == -1))"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)

def create_rl_notebook(output_path):
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ChroniCare: Reinforcement Learning Recommendation Engine\n",
                    "This notebook describes the discrete **Q-Learning agent** which serves as ChroniCare's personalized treatment and lifestyle advisor.\n",
                    "\n",
                    "### RL Framing:\n",
                    "- **State space**: Discretized indices of Glucose (0: Normal, 1: Prediabetic, 2: Diabetic), BMI (0: Normal, 1: Overweight, 2: Obese), and Blood Pressure (0: Normal, 1: Prehypertension, 2: Hypertension). Total states = 27.\n",
                    "- **Action space**: 4 lifestyle recommendations:\n",
                    "  - Action 0: Carb Reduction\n",
                    "  - Action 1: Adjust Exercise\n",
                    "  - Action 2: Monitor Vitals\n",
                    "  - Action 3: Professional Consultation\n",
                    "- **Reward function**: Modeled clinically based on lifestyle actions moving patients toward healthier vital thresholds."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "\n",
                    "n_states = 27\n",
                    "n_actions = 4\n",
                    "q_table = np.zeros((n_states, n_actions))\n",
                    "\n",
                    "def encode_state(g, bmi, bp):\n",
                    "    return g * 9 + bmi * 3 + bp\n",
                    "\n",
                    "def decode_state(state):\n",
                    "    g = state // 9\n",
                    "    bmi = (state % 9) // 3\n",
                    "    bp = state % 3\n",
                    "    return g, bmi, bp"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Q-Learning Training Algorithm"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "alpha = 0.1\n",
                    "gamma = 0.9\n",
                    "epsilon = 0.3\n",
                    "episodes = 5000\n",
                    "\n",
                    "for ep in range(episodes):\n",
                    "    # Random starting state\n",
                    "    g, bmi, bp = np.random.choice([0,1,2]), np.random.choice([0,1,2]), np.random.choice([0,1,2])\n",
                    "    state = encode_state(g, bmi, bp)\n",
                    "    done = False\n",
                    "    steps = 0\n",
                    "    \n",
                    "    while not done and steps < 10:\n",
                    "        # Action selection (epsilon-greedy)\n",
                    "        if np.random.rand() < epsilon:\n",
                    "            action = np.random.choice(n_actions)\n",
                    "        else:\n",
                    "            action = np.argmax(q_table[state])\n",
                    "            \n",
                    "        next_g, next_bmi, next_bp = g, bmi, bp\n",
                    "        reward = 0\n",
                    "        \n",
                    "        if action == 0:\n",
                    "            if g > 0:\n",
                    "                next_g = max(0, g - 1)\n",
                    "                reward += 15\n",
                    "            else: reward += 5\n",
                    "        elif action == 1:\n",
                    "            if bmi > 0:\n",
                    "                next_bmi = max(0, bmi - 1)\n",
                    "                reward += 15\n",
                    "            if bp > 0:\n",
                    "                next_bp = max(0, bp - 1)\n",
                    "                reward += 10\n",
                    "            else: reward += 5\n",
                    "        elif action == 2:\n",
                    "            reward += 8\n",
                    "        elif action == 3:\n",
                    "            if g == 2 or bmi == 2 or bp == 2:\n",
                    "                next_g = max(0, g - 1)\n",
                    "                next_bp = max(0, bp - 1)\n",
                    "                reward += 20\n",
                    "            else: reward += 5\n",
                    "            \n",
                    "        reward -= (next_g * 6 + next_bmi * 4 + next_bp * 5)\n",
                    "        next_state = encode_state(next_g, next_bmi, next_bp)\n",
                    "        \n",
                    "        # Q-learning Bellman Update\n",
                    "        q_table[state, action] = q_table[state, action] + alpha * (\n",
                    "            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]\n",
                    "        )\n",
                    "        \n",
                    "        g, bmi, bp = next_g, next_bmi, next_bp\n",
                    "        state = next_state\n",
                    "        steps += 1\n",
                    "        if g == 0 and bmi == 0 and bp == 0:\n",
                    "            done = True\n",
                    "\n",
                    "print(\"Learned Q-table (States x Actions):\")\n",
                    "print(q_table[:5])"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)

def main():
    notebooks_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\notebooks"
    os.makedirs(notebooks_dir, exist_ok=True)
    
    print("Writing supervised_training.ipynb...")
    create_supervised_notebook(os.path.join(notebooks_dir, "supervised_training.ipynb"))
    
    print("Writing unsupervised_training.ipynb...")
    create_unsupervised_notebook(os.path.join(notebooks_dir, "unsupervised_training.ipynb"))
    
    print("Writing rl_training.ipynb...")
    create_rl_notebook(os.path.join(notebooks_dir, "rl_training.ipynb"))
    
    print("Notebook generation finished successfully!")

if __name__ == "__main__":
    main()
