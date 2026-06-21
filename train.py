import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Supervised models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

# Unsupervised models
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

def train_supervised_models(data_dir, models_dir):
    metrics = {}
    
    # --- 1. DIABETES (Random Forest & Logistic Regression) ---
    print("Training Diabetes models...")
    df = pd.read_csv(os.path.join(data_dir, "diabetes.csv"))
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    scaler_diab = StandardScaler()
    X_scaled = scaler_diab.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    
    # Save
    with open(os.path.join(models_dir, "diabetes_rf.pkl"), "wb") as f:
        pickle.dump((rf, scaler_diab), f)
    with open(os.path.join(models_dir, "diabetes_lr.pkl"), "wb") as f:
        pickle.dump((lr, scaler_diab), f)
        
    metrics['Diabetes_RF'] = calculate_metrics(y_test, y_pred_rf, y_prob_rf)
    metrics['Diabetes_LR'] = calculate_metrics(y_test, y_pred_lr, y_prob_lr)
    
    # --- 2. HEART DISEASE (SVM & Decision Tree) ---
    print("Training Heart Disease models...")
    df = pd.read_csv(os.path.join(data_dir, "heart.csv"))
    X = df.drop(columns=['target'])
    y = df['target']
    
    scaler_heart = StandardScaler()
    X_scaled = scaler_heart.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # SVM
    svm = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]
    
    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1]
    
    # Save
    with open(os.path.join(models_dir, "heart_svm.pkl"), "wb") as f:
        pickle.dump((svm, scaler_heart), f)
    with open(os.path.join(models_dir, "heart_dt.pkl"), "wb") as f:
        pickle.dump((dt, scaler_heart), f)
        
    metrics['Heart_SVM'] = calculate_metrics(y_test, y_pred_svm, y_prob_svm)
    metrics['Heart_DT'] = calculate_metrics(y_test, y_pred_dt, y_prob_dt)
    
    # --- 3. KIDNEY DISEASE (KNN & Naive Bayes) ---
    print("Training Kidney Disease models...")
    df = pd.read_csv(os.path.join(data_dir, "kidney.csv"))
    X = df.drop(columns=['class'])
    y = df['class']
    
    scaler_kidney = StandardScaler()
    X_scaled = scaler_kidney.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_prob_knn = knn.predict_proba(X_test)[:, 1]
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    y_prob_nb = nb.predict_proba(X_test)[:, 1]
    
    # Save
    with open(os.path.join(models_dir, "kidney_knn.pkl"), "wb") as f:
        pickle.dump((knn, scaler_kidney), f)
    with open(os.path.join(models_dir, "kidney_nb.pkl"), "wb") as f:
        pickle.dump((nb, scaler_kidney), f)
        
    metrics['Kidney_KNN'] = calculate_metrics(y_test, y_pred_knn, y_prob_knn)
    metrics['Kidney_NB'] = calculate_metrics(y_test, y_pred_nb, y_prob_nb)
    
    # --- 4. LIVER DISEASE (XGBoost & Gradient Boosting) ---
    print("Training Liver Disease models...")
    df = pd.read_csv(os.path.join(data_dir, "liver.csv"))
    X = df.drop(columns=['Dataset'])
    y = df['Dataset']
    
    scaler_liver = StandardScaler()
    X_scaled = scaler_liver.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # XGBoost
    xgb = XGBClassifier(eval_metric='logloss', max_depth=4, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(max_depth=4, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    y_prob_gb = gb.predict_proba(X_test)[:, 1]
    
    # Save
    with open(os.path.join(models_dir, "liver_xgb.pkl"), "wb") as f:
        pickle.dump((xgb, scaler_liver), f)
    with open(os.path.join(models_dir, "liver_gb.pkl"), "wb") as f:
        pickle.dump((gb, scaler_liver), f)
        
    metrics['Liver_XGB'] = calculate_metrics(y_test, y_pred_xgb, y_prob_xgb)
    metrics['Liver_GB'] = calculate_metrics(y_test, y_pred_gb, y_prob_gb)
    
    # Save global metrics
    with open(os.path.join(models_dir, "model_metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    print("Supervised training complete!")

def calculate_metrics(y_true, y_pred, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': roc_auc
    }

def train_unsupervised_models(data_dir, models_dir):
    print("Training Unsupervised models (K-Means, PCA, DBSCAN)...")
    df = pd.read_csv(os.path.join(data_dir, "diabetes.csv"))
    # Select feature columns (excluding Outcome)
    features = df.drop(columns=['Outcome'])
    
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(features)
    
    # PCA
    pca = PCA(n_components=3, random_state=42)
    pca_coords = pca.fit_transform(scaled_feats)
    
    # K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_feats)
    
    # DBSCAN
    dbscan = DBSCAN(eps=1.8, min_samples=5)
    dbscan_labels = dbscan.fit_predict(scaled_feats)
    
    # Save clustering & dimensional results
    with open(os.path.join(models_dir, "kmeans_diabetes.pkl"), "wb") as f:
        pickle.dump((kmeans, scaler), f)
        
    with open(os.path.join(models_dir, "pca_diabetes.pkl"), "wb") as f:
        pickle.dump(pca, f)
        
    # We serialize K-Means and DBSCAN coordinates/labels for visualization
    dbscan_data = {
        'scaled_features': scaled_feats,
        'pca_coords': pca_coords,
        'kmeans_labels': cluster_labels,
        'dbscan_labels': dbscan_labels,
        'original_features': features
    }
    with open(os.path.join(models_dir, "dbscan_diabetes.pkl"), "wb") as f:
        pickle.dump(dbscan_data, f)
        
    print("Unsupervised training complete!")

def train_rl_agent(models_dir):
    print("Training Reinforcement Learning (Q-Learning) Treatment Advisor...")
    # State space size: 3 (Glucose) * 3 (BMI) * 3 (BP) = 27 states
    # Action space size: 4 actions:
    # 0: Carb Reduction
    # 1: Adjust Exercise
    # 2: Monitor Vitals
    # 3: Professional Consultation
    
    n_states = 27
    n_actions = 4
    
    q_table = np.zeros((n_states, n_actions))
    
    # Hyperparameters
    alpha = 0.1      # Learning rate
    gamma = 0.9      # Discount factor
    epsilon = 0.3    # Exploration rate
    episodes = 8000
    
    # Helper to encode state: g in [0, 1, 2], bmi in [0, 1, 2], bp in [0, 1, 2]
    def encode_state(g, bmi, bp):
        return g * 9 + bmi * 3 + bp

    # Helper to decode state
    def decode_state(state):
        g = state // 9
        bmi = (state % 9) // 3
        bp = state % 3
        return g, bmi, bp
    
    np.random.seed(46)
    
    for episode in range(episodes):
        # Initial random state
        g = np.random.choice([0, 1, 2])
        bmi = np.random.choice([0, 1, 2])
        bp = np.random.choice([0, 1, 2])
        state = encode_state(g, bmi, bp)
        
        done = False
        steps = 0
        
        while not done and steps < 10:
            # Action selection (epsilon-greedy)
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                action = np.argmax(q_table[state])
            
            # Next state simulation and reward logic
            next_g, next_bmi, next_bp = g, bmi, bp
            reward = 0
            
            if action == 0:  # Carb Reduction
                if g > 0:
                    next_g = max(0, g - 1)
                    reward += 15  # Good action
                else:
                    reward += 5   # Neutral/healthy
            elif action == 1:  # Exercise
                if bmi > 0:
                    next_bmi = max(0, bmi - 1)
                    reward += 15
                if bp > 0:
                    next_bp = max(0, bp - 1)
                    reward += 10
                else:
                    reward += 5
            elif action == 2:  # Monitor Vitals
                # Vitals monitoring ensures no further state deterioration, yields minor reward
                reward += 8
            elif action == 3:  # Clinical Consultation
                # High benefit, especially in higher states, but minor resource cost
                if g == 2 or bmi == 2 or bp == 2:
                    next_g = max(0, g - 1)
                    next_bp = max(0, bp - 1)
                    reward += 20
                else:
                    reward += 5
            
            # Penalize bad biological states
            reward -= (next_g * 6 + next_bmi * 4 + next_bp * 5)
            
            next_state = encode_state(next_g, next_bmi, next_bp)
            
            # Bellman equation update
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            
            # Update state
            g, bmi, bp = next_g, next_bmi, next_bp
            state = next_state
            steps += 1
            
            # Stop if state is optimal (all 0s)
            if g == 0 and bmi == 0 and bp == 0:
                done = True
                
    # Save the Q-table
    np.save(os.path.join(models_dir, "q_table.npy"), q_table)
    print("Q-learning training complete!")

def main():
    data_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\data"
    models_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\models"
    
    os.makedirs(models_dir, exist_ok=True)
    
    train_supervised_models(data_dir, models_dir)
    train_unsupervised_models(data_dir, models_dir)
    train_rl_agent(models_dir)
    print("All models successfully trained and serialized to models/ directory!")

if __name__ == "__main__":
    main()
