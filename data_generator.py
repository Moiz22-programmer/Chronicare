import os
import pandas as pd
import numpy as np

def generate_diabetes_data(n_samples=1000):
    np.random.seed(42)
    # Generate features with realistic clinical distributions
    pregnancies = np.random.poisson(lam=2.5, size=n_samples)
    pregnancies = np.clip(pregnancies, 0, 17)
    
    glucose = np.random.normal(loc=115, scale=30, size=n_samples)
    glucose = np.clip(glucose, 44, 200)
    
    blood_pressure = np.random.normal(loc=72, scale=12, size=n_samples)
    blood_pressure = np.clip(blood_pressure, 40, 122)
    
    skin_thickness = np.random.normal(loc=25, scale=10, size=n_samples)
    skin_thickness = np.clip(skin_thickness, 7, 99)
    
    insulin = np.random.exponential(scale=120, size=n_samples) + 15
    insulin = np.clip(insulin, 14, 846)
    
    bmi = np.random.normal(loc=32, scale=6.5, size=n_samples)
    bmi = np.clip(bmi, 18.2, 67.1)
    
    diabetes_pedigree = np.random.beta(a=2, b=5, size=n_samples) * 2.5
    diabetes_pedigree = np.clip(diabetes_pedigree, 0.08, 2.42)
    
    age = np.random.normal(loc=33, scale=11, size=n_samples)
    age = np.clip(age, 21, 81).astype(int)
    
    # Calculate probability of diabetes based on clinical risk factors
    z = (0.04 * (glucose - 100) + 
         0.08 * (bmi - 25) + 
         0.03 * (age - 30) + 
         0.5 * diabetes_pedigree + 
         0.05 * pregnancies + 
         0.002 * (insulin - 80) - 2.5)
    
    prob = 1 / (1 + np.exp(-z))
    outcome = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose.astype(int),
        'BloodPressure': blood_pressure.astype(int),
        'SkinThickness': skin_thickness.astype(int),
        'Insulin': insulin.astype(int),
        'BMI': np.round(bmi, 1),
        'DiabetesPedigreeFunction': np.round(diabetes_pedigree, 3),
        'Age': age,
        'Outcome': outcome
    })
    return df

def generate_heart_data(n_samples=1000):
    np.random.seed(43)
    age = np.random.normal(loc=54, scale=9, size=n_samples)
    age = np.clip(age, 29, 77).astype(int)
    
    sex = np.random.binomial(n=1, p=0.68, size=n_samples) # 1=male, 0=female
    
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.47, 0.17, 0.28, 0.08]) # chest pain type
    
    trestbps = np.random.normal(loc=131, scale=17, size=n_samples)
    trestbps = np.clip(trestbps, 94, 200).astype(int)
    
    chol = np.random.normal(loc=246, scale=50, size=n_samples)
    chol = np.clip(chol, 126, 564).astype(int)
    
    fbs = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]) # fasting blood sugar > 120
    
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.49, 0.48, 0.03])
    
    # max heart rate thalach is inversely correlated with age
    thalach_loc = 200 - 0.7 * age
    thalach = np.random.normal(loc=thalach_loc, scale=20, size=n_samples)
    thalach = np.clip(thalach, 71, 202).astype(int)
    
    exang = np.random.binomial(n=1, p=0.33, size=n_samples) # exercise induced angina
    
    oldpeak = np.random.exponential(scale=1.0, size=n_samples)
    oldpeak = np.clip(oldpeak, 0.0, 6.2)
    
    slope = np.random.choice([0, 1, 2], size=n_samples, p=[0.07, 0.46, 0.47])
    
    ca = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.57, 0.22, 0.13, 0.06, 0.02]) # vessels colored
    
    thal = np.random.choice([1, 2, 3], size=n_samples, p=[0.06, 0.54, 0.40]) # thalassemia
    
    # z score calculation for heart disease
    z = (0.05 * (age - 50) + 
         0.8 * sex + 
         0.6 * (cp == 0) + # asymptomatic CP often correlated with higher severity/risk in Cleveland
         0.02 * (trestbps - 120) + 
         0.005 * (chol - 200) - 
         0.03 * (thalach - 150) + 
         1.0 * exang + 
         0.8 * oldpeak + 
         0.7 * ca + 
         0.6 * (thal == 3) - 2.8)
    
    prob = 1 / (1 + np.exp(-z))
    target = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': np.round(oldpeak, 1),
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'target': target
    })
    return df

def generate_kidney_data(n_samples=1000):
    np.random.seed(44)
    age = np.random.normal(loc=51, scale=17, size=n_samples)
    age = np.clip(age, 2, 90).astype(int)
    
    bp = np.random.normal(loc=76, scale=13, size=n_samples)
    bp = np.clip(bp, 50, 180).astype(int)
    
    sg = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], size=n_samples, p=[0.08, 0.22, 0.20, 0.38, 0.12])
    
    al = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.60, 0.12, 0.10, 0.08, 0.06, 0.04]) # albumin
    su = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.85, 0.05, 0.03, 0.03, 0.02, 0.02]) # sugar
    
    rbc = np.random.choice([1, 0], size=n_samples, p=[0.80, 0.20]) # normal=1, abnormal=0
    pc = np.random.choice([1, 0], size=n_samples, p=[0.84, 0.16])  # normal=1, abnormal=0
    pcc = np.random.choice([0, 1], size=n_samples, p=[0.89, 0.11]) # notpresent=0, present=1
    ba = np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08])  # notpresent=0, present=1
    
    bgr = np.random.normal(loc=148, scale=79, size=n_samples)
    bgr = np.clip(bgr, 22, 490).astype(int)
    
    bu = np.random.normal(loc=57, scale=50, size=n_samples)
    bu = np.clip(bu, 1.5, 391).astype(int)
    
    sc = np.random.exponential(scale=3.0, size=n_samples) + 0.4
    sc = np.clip(sc, 0.4, 32.0)
    
    sod = np.random.normal(loc=137, scale=10, size=n_samples)
    sod = np.clip(sod, 4.5, 163).astype(int)
    
    pot = np.random.normal(loc=4.6, scale=3.1, size=n_samples)
    pot = np.clip(pot, 2.5, 47.0)
    
    hemo = np.random.normal(loc=12.5, scale=2.9, size=n_samples)
    hemo = np.clip(hemo, 3.1, 17.8)
    
    pcv = (hemo * 3.1 + np.random.normal(loc=0, scale=2, size=n_samples)).astype(int)
    pcv = np.clip(pcv, 9, 54)
    
    wc = np.random.normal(loc=8400, scale=2900, size=n_samples)
    wc = np.clip(wc, 2200, 26400).astype(int)
    
    rc = np.random.normal(loc=4.7, scale=1.0, size=n_samples)
    rc = np.clip(rc, 2.1, 8.0)
    
    htn = np.random.binomial(n=1, p=0.37, size=n_samples)
    dm = np.random.binomial(n=1, p=0.34, size=n_samples)
    cad = np.random.binomial(n=1, p=0.08, size=n_samples)
    appet = np.random.choice([1, 0], size=n_samples, p=[0.79, 0.21]) # good=1, poor=0
    pe = np.random.binomial(n=1, p=0.19, size=n_samples)
    ane = np.random.binomial(n=1, p=0.15, size=n_samples)
    
    z = (0.02 * (age - 50) + 
         0.03 * (bp - 80) - 
         40.0 * (sg - 1.020) + 
         1.2 * al + 
         0.5 * su + 
         1.5 * (1 - rbc) + 
         1.0 * (1 - pc) + 
         0.005 * (bgr - 120) + 
         0.01 * bu + 
         0.8 * sc - 
         0.4 * (hemo - 12) + 
         1.8 * htn + 
         1.5 * dm + 
         1.0 * (1 - appet) + 
         1.2 * pe - 2.0)
    
    prob = 1 / (1 + np.exp(-z))
    class_label = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'age': age,
        'bp': bp,
        'sg': sg,
        'al': al,
        'su': su,
        'rbc': rbc,
        'pc': pc,
        'pcc': pcc,
        'ba': ba,
        'bgr': bgr,
        'bu': bu,
        'sc': np.round(sc, 1),
        'sod': sod,
        'pot': np.round(pot, 1),
        'hemo': np.round(hemo, 1),
        'pcv': pcv,
        'wc': wc,
        'rc': np.round(rc, 1),
        'htn': htn,
        'dm': dm,
        'cad': cad,
        'appet': appet,
        'pe': pe,
        'ane': ane,
        'class': class_label
    })
    return df

def generate_liver_data(n_samples=1000):
    np.random.seed(45)
    age = np.random.normal(loc=45, scale=16, size=n_samples)
    age = np.clip(age, 4, 90).astype(int)
    
    gender = np.random.choice([1, 0], size=n_samples, p=[0.75, 0.25]) # 1=Male, 0=Female
    
    total_bilirubin = np.random.exponential(scale=3.0, size=n_samples) + 0.4
    total_bilirubin = np.clip(total_bilirubin, 0.4, 75.0)
    
    # direct bilirubin is highly correlated with total bilirubin
    direct_bilirubin = total_bilirubin * 0.45 + np.random.normal(loc=0, scale=0.1, size=n_samples)
    direct_bilirubin = np.clip(direct_bilirubin, 0.1, 19.7)
    
    alkphos = np.random.normal(loc=290, scale=240, size=n_samples)
    alkphos = np.clip(alkphos, 63, 2110).astype(int)
    
    sgpt = np.random.exponential(scale=80, size=n_samples) + 10
    sgpt = np.clip(sgpt, 10, 2000).astype(int) # ALT
    
    sgot = sgpt * 1.8 + np.random.normal(loc=0, scale=20, size=n_samples)
    sgot = np.clip(sgot, 10, 4929).astype(int) # AST
    
    total_proteins = np.random.normal(loc=6.5, scale=1.1, size=n_samples)
    total_proteins = np.clip(total_proteins, 2.7, 9.6)
    
    albumin = np.random.normal(loc=3.1, scale=0.8, size=n_samples)
    albumin = np.clip(albumin, 0.9, 5.5)
    
    ag_ratio = albumin / (total_proteins - albumin)
    ag_ratio = np.nan_to_num(ag_ratio, nan=0.9)
    ag_ratio = np.clip(ag_ratio, 0.3, 2.8)
    
    z = (0.01 * (age - 40) + 
         0.3 * gender + 
         0.8 * total_bilirubin + 
         1.2 * direct_bilirubin + 
         0.002 * (alkphos - 200) + 
         0.008 * sgpt + 
         0.002 * sgot - 
         0.5 * albumin - 
         1.1 * ag_ratio - 0.5)
    
    prob = 1 / (1 + np.exp(-z))
    dataset = (np.random.rand(n_samples) < prob).astype(int) # 1=liver patient, 0=healthy
    
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Total_Bilirubin': np.round(total_bilirubin, 1),
        'Direct_Bilirubin': np.round(direct_bilirubin, 1),
        'Alkaline_Phosphotase': alkphos,
        'Alamine_Aminotransferase': sgpt,
        'Aspartate_Aminotransferase': sgot,
        'Total_Protiens': np.round(total_proteins, 1),
        'Albumin': np.round(albumin, 1),
        'Albumin_and_Globulin_Ratio': np.round(ag_ratio, 2),
        'Dataset': dataset
    })
    return df

def main():
    data_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating Diabetes (PIMA) dataset...")
    diabetes_df = generate_diabetes_data()
    diabetes_df.to_csv(os.path.join(data_dir, "diabetes.csv"), index=False)
    
    print("Generating Heart Disease (Cleveland) dataset...")
    heart_df = generate_heart_data()
    heart_df.to_csv(os.path.join(data_dir, "heart.csv"), index=False)
    
    print("Generating Chronic Kidney Disease (CKD) dataset...")
    kidney_df = generate_kidney_data()
    kidney_df.to_csv(os.path.join(data_dir, "kidney.csv"), index=False)
    
    print("Generating Liver Patient (ILPD) dataset...")
    liver_df = generate_liver_data()
    liver_df.to_csv(os.path.join(data_dir, "liver.csv"), index=False)
    
    print("All datasets generated successfully!")

if __name__ == "__main__":
    main()
