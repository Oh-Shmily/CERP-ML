from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.base import clone
from xgboost import XGBRegressor
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from utils import dna_to_onehot

forecast_lindel_train_path = '/data/combine_forecast_lindel_data/forecast_lindel_train.csv'
forecast_lindel_valid_path = '/data/combine_forecast_lindel_data/forecast_lindel_valid.csv'
train_path = '/data/forecast_train_val_test/train.csv'
valid_path = '/data/forecast_train_val_test/valid.csv'
test_path = '/data/forecast_train_val_test/test.csv'
lindel_test_path = '/data/compiled_lindel_data/lindel_test.csv'
model_save_path = '/machine_learning/model_param/bset_reg.pkl'

def train_valid(train_path, valid_path, model, model_save_path):
    # train
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    # split
    X_train = dna_to_onehot(train_df['target_seq'])
    y_train = train_df.iloc[:, -557:]
    X_valid = dna_to_onehot(valid_df['target_seq'])
    y_valid = valid_df.iloc[:, -557:]

    regressor = model

    regressor.fit(X_train, y_train, eval_metric='mphe', verbose=True, eval_set=[(X_valid, y_valid)])

    joblib.dump(regressor, model_save_path)


def test(test_path, model_save_path):
    # test
    forecast_test_df = pd.read_csv(test_path[0])
    lindel_test_df = pd.read_csv(test_path[1])
    X_test = dna_to_onehot(forecast_test_df['target_seq'])
    y_test = forecast_test_df.iloc[:, -557:]
    X_lindel = dna_to_onehot(lindel_test_df['target_seq'])
    y_lindel = lindel_test_df.iloc[:, -557:]

    regressor = joblib.load(model_save_path)
    y_pred = regressor.predict(X_test)
    lindel_pred = regressor.predict(X_lindel)

    mse = mean_squared_error(y_test, y_pred)
    lindel_mse = mean_squared_error(y_lindel, lindel_pred)

    print(f'forecast mse: {mse}, lindel mse: {lindel_mse}')


def models_cross_validation(train_path, valid_path, model, n_splits, k_mer=1, step=1):
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    train_df = pd.concat([train_df, valid_df], axis=0)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold = 0
    mse_scores = []
    pearson_scores = []

    for train_index, valid_index in kf.split(train_df):
        fold += 1
        print(f"Training fold {fold}...")

        # Split the data
        train_fold_df = train_df.iloc[train_index]
        valid_fold_df = train_df.iloc[valid_index]
        
        X_train_fold = dna_to_onehot(train_fold_df['target_seq'], k_mer=k_mer, step=step)
        y_train_fold = train_fold_df.iloc[:, -557:]
        X_valid_fold = dna_to_onehot(valid_fold_df['target_seq'], k_mer=k_mer, step=step)
        y_valid_fold = valid_fold_df.iloc[:, -557:]
        
        # Clone the model to ensure a fresh start for each fold
        regressor = clone(model)
        
        regressor.fit(X_train_fold, y_train_fold)
        
        # Predict on validation fold
        y_valid_pred = regressor.predict(X_valid_fold)
        valid_mse = mean_squared_error(y_valid_fold, y_valid_pred)
        print(f"Fold {fold} MSE: {valid_mse}")

        # Aggregate the results
        mse_scores.append(valid_mse)

    return mse_scores, pearson_scores

def compare_models(train_path, valid_path, models, n_splits=10):
    all_mse_scores = []
    all_pearson_scores = []
    all_model = []

    for model in models:
        model_name = type(model).__name__
        print(f"Evaluating model: {model_name}")
        
        mse_scores, pearson_scores = models_cross_validation(train_path, valid_path, model, n_splits=n_splits)
        all_mse_scores.extend(mse_scores)
        all_pearson_scores.extend(pearson_scores)
        all_model.extend([model_name] * n_splits)
    
    # Create a DataFrame for plotting
    results_df = pd.DataFrame({
        'Model': all_model,
        'MSE': all_mse_scores,
        'PEARSON': all_pearson_scores
    })
    
    results_df.to_csv('/machine_learning/compare_results.csv', index=False)



models = [
    KNeighborsRegressor(),
    RandomForestRegressor(),
    ElasticNet(),
    XGBRegressor()
]

compare_models(forecast_lindel_train_path, forecast_lindel_valid_path, models)

model = XGBRegressor(objective='reg:logistic', reg_alpha=0.2, reg_lambda=0.8)

train_valid(forecast_lindel_train_path, forecast_lindel_valid_path, model, model_save_path)
test([test_path, lindel_test_path], model_save_path)
