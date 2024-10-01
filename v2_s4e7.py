# import dependencies
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# import seaborn as sns
import gc
import warnings
warnings.simplefilter('ignore')


# import data
train = pd.read_csv('./playground-series-s4e7/train.csv')
train = train.drop('id', axis = 1)
test = pd.read_csv('./playground-series-s4e7/test.csv')
submission = test[['id']]
test = test.drop('id', axis = 1)

# preprocess

# preprocess -- step 1: modify data type of columns
train_df = train.copy()
targets = train['Response']
train = train.drop('Response', axis = 1)
df_total = pd.concat([train, test])

def preprocess(data):
    df = data.copy()
    # modify the data type to reduce memory usage.        
    df['Driving_License'] = df['Driving_License'].astype('int8')
    df['Previously_Insured'] = df['Previously_Insured'].astype('int8')
    df['Region_Code'] = df['Region_Code'].astype('int16')
    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype('int16')
    df['Annual_Premium'] = df['Annual_Premium'].astype('int32')
    # change the object data types into the int types   
    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype('int8')
    df['Vehicle_Age'] = df['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}).astype('int8')
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'No': 0, 'Yes': 1}).astype('int8')

    return df

df_total = preprocess(df_total)

# preprocess -- step 2: feature engineering
# df_total['Prev_Insured_Vehicle_Age'] = pd.factorize(
#         (df_total['Previously_Insured'].astype(str) + df_total['Vehicle_Age'].astype(str)).to_numpy()
#     )[0]

df_total['Prev_Insured_Vehicle_Damage'] = pd.factorize(
        (df_total['Previously_Insured'].astype(str) + df_total['Vehicle_Damage'].astype(str)).to_numpy()
    )[0]
# preprocess -- step 3: aggregation encoding
# ex = df_total.groupby('Policy_Sales_Channel').agg({"Age": ["mean"]})
# ex.reset_index(inplace=True)
# ex.columns = ['Policy_Sales_Channel', "PSC_Age_mean"]
# ex['PSC_Age_mean'] = ex['PSC_Age_mean'].astype('int32')
# df_total = df_total.merge(ex, on="Policy_Sales_Channel", how='left')

# preprocess -- step 4: target_encoding
age_encoding = pd.pivot_table(data = train_df, index = 'Age', values = 'Response', aggfunc = 'sum')
age_encoding.reset_index(inplace=True)
age_encoding.columns = ['Age_encoding', 'values']
replace_dict = age_encoding.to_dict()
df_total['Age_encoding'] = df_total['Age']
df_total = df_total.replace(replace_dict)

# split into train and test
train = df_total[:train.shape[0]]
test = df_total[train.shape[0]:]

# clear cache memory
X_train = train.copy()
y_train = targets.copy()
X_test = test.copy()

del train, test, targets, df_total, train_df#, ex
gc.collect()

# CatBoost modeling and training
cv = StratifiedKFold(5, shuffle=True, random_state=0)
cv_splits = cv.split(X_train, y_train)
test_preds = list()
params = {
    'nan_mode': 'Min',
    'gpu_ram_part': 0.85,
    'eval_metric': 'AUC',
    'combinations_ctr': ['Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1',
    'FeatureFreq:CtrBorderCount=15:CtrBorderType=Median:Prior=0/1'],
    'iterations': 7000,
    'fold_permutation_block': 64,
    'leaf_estimation_method': 'Newton',
    'od_pval': 0,
    'random_score_type': 'NormalWithModelSizeDecrease',
    'counter_calc_method': 'SkipTest',
    'grow_policy': 'SymmetricTree',
    'penalties_coefficient': 1,
    'boosting_type': 'Plain',
    'ctr_history_unit': 'Sample',
    'feature_border_type': 'GreedyLogSum',
    'one_hot_max_size': 2,
    'devices': '-1',
    'eval_fraction': 0,
    'l2_leaf_reg': 0.5,
    'random_strength': 0,
    'od_type': 'Iter',
    'rsm': 1,
    'boost_from_average': False,
    'gpu_cat_features_storage': 'GpuRam',
    'max_ctr_complexity': 4,
    'model_size_reg': 0.5,
    'simple_ctr': ['Borders:CtrBorderCount=15:CtrBorderType=Uniform:TargetBorderCount=1:TargetBorderType=MinEntropy:Prior=0/1:Prior=0.5/1:Prior=1/1',
    'FeatureFreq:CtrBorderCount=15:CtrBorderType=MinEntropy:Prior=0/1'],
    'use_best_model': True,
    'od_wait': 200,
    'class_names': [0, 1],
    'random_seed': 3157,
    'depth': 9,
    'ctr_target_border_count': 1,
    'has_time': False,
    'border_count': 128,
    'data_partition': 'FeatureParallel',
    'bagging_temperature': 1,
    'classes_count': 0,
    'auto_class_weights': 'None',
    'leaf_estimation_backtracking': 'AnyImprovement',
    'best_model_min_trees': 1,
    'min_data_in_leaf': 1,
    'loss_function': 'Logloss',
    'learning_rate': 0.073,
    'score_function': 'Cosine',
    'task_type': 'GPU',
    'leaf_estimation_iterations': 10,
    'bootstrap_type': 'Bayesian',
    'max_leaves': 512,
}

for i, (train_idx, val_idx) in enumerate(cv_splits):
    model = CatBoostClassifier(**params, verbose=False)
    X_train_fold, X_val_fold = X_train.loc[train_idx], X_train.loc[val_idx]
    y_train_fold, y_val_fold = y_train.loc[train_idx], y_train.loc[val_idx]
    X_train_pool = Pool(X_train, y_train, cat_features=X_train.columns.values)
    X_valid_pool = Pool(X_val_fold, y_val_fold, cat_features=X_val_fold.columns.values)
    X_test_pool = Pool(X_test[X_train.columns], cat_features=X_test.columns.values)
    model.fit(X=X_train_pool, eval_set=X_valid_pool, verbose=1000, early_stopping_rounds=200)
    test_pred = model.predict_proba(X_test_pool)[:, 1]
    test_preds.append(test_pred)
    del X_train_fold, X_val_fold, y_train_fold, y_val_fold
    del X_train_pool, X_valid_pool, X_test_pool
    del model, test_pred
    gc.collect()
    print(f'Fold {i+1} finished.')

submission['Response'] = np.mean(test_preds, axis=0)
submission.to_csv('submission.csv', index=False)


# catb_params = {
#     'task_type': 'GPU',
#     'loss_function': 'Logloss',
#     'eval_metric': 'AUC',
#     'bootstrap_type': 'Bayesian',
#     'grow_policy': 'Lossguide',
#     'iterations': 10000,
#     'learning_rate': 0.05,
#     'thread_count': 4,
#     'verbose': 500,
#     'num_leaves': 267,
#     'bagging_temperature': 0.22080996289544302,
#     'depth': 7,
#     'border_count': 994,
#     'min_child_samples': 93,
#     'random_strength': 0.7218648648351215,
#     'l2_leaf_reg': 80.74974748224912,
#     'model_size_reg': 0.5426723735193903,
#     'random_seed': 1001,
# }