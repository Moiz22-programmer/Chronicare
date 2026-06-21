import pickle, os
import pandas as pd
models_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\models"
data_dir = r"c:\Users\AL RAHMAN LAPTOP\OneDrive\Desktop\ML\ChroniCare\data"

p = os.path.join(models_dir, 'heart_dt.pkl')
if not os.path.exists(p):
    print('MODEL_NOT_FOUND')
    raise SystemExit(1)

with open(p, 'rb') as f:
    dt, scaler = pickle.load(f)

# load heart.csv to get feature names
df = pd.read_csv(os.path.join(data_dir, 'heart.csv'))
feature_names = list(df.drop(columns=['target']).columns)

# Access tree structure
tree = dt.tree_
root_feature_index = tree.feature[0]
root_threshold = tree.threshold[0]

print('root_feature_index=', root_feature_index)
print('root_feature_name=', feature_names[root_feature_index])
print('root_threshold=', root_threshold)
print('criterion=', getattr(dt, 'criterion', 'gini (default)'))
print('max_depth=', getattr(dt, 'max_depth', None))
print('n_features=', dt.n_features_in_)
print('feature_importances=', dt.feature_importances_)
