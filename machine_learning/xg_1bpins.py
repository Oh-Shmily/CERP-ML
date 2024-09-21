from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr, kendalltau
warnings.filterwarnings("ignore")

from evaluate import get_eval
from utils import dna_to_onehot

forecast_lindel_train_path = '/data/combine_forecast_lindel_data/forecast_lindel_train.csv'
forecast_lindel_valid_path = '/data/combine_forecast_lindel_data/forecast_lindel_valid.csv'
train_path = '/data/forecast_train_val_test/train.csv'
valid_path = '/data/forecast_train_val_test/valid.csv'
test_path = '/data/forecast_train_val_test/test.csv'
lindel_test_path = '/data/compiled_lindel_data/lindel_test.csv'
sprout_test_path = '/data/new_key_df.csv'
model_save_path = '/machine_learning/model_param/xg_1bp_ins.pkl'

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

def train_valid(train_path, valid_path, model, model_save_path):
    # train
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    # split
    X_train = dna_to_onehot(train_df['target_seq'])
    y_train = np.sum(train_df.iloc[:, -21:-17], axis=1)
    X_valid = dna_to_onehot(valid_df['target_seq'])
    y_valid = np.sum(valid_df.iloc[:, -21:-17], axis=1)

    regressor = model

    regressor.fit(X_train, y_train, eval_metric='mphe', verbose=True, eval_set=[(X_valid, y_valid)])

    joblib.dump(regressor, model_save_path)


cor_list = []
k_list = []
def test(test_path, model_save_path, seed):
    # test
    forecast_test_df = pd.read_csv(test_path[0])
    lindel_test_df = pd.read_csv(test_path[1])
    sprout_test_df = pd.read_csv(test_path[2])
    sprout_test_df = sprout_test_df.sample(n=142, random_state=seed)
    X_test = dna_to_onehot(forecast_test_df['target_seq'])
    y_test = np.sum(forecast_test_df.iloc[:, -21:-17], axis=1)
    X_lindel = dna_to_onehot(lindel_test_df['target_seq'])
    y_lindel = np.sum(lindel_test_df.iloc[:, -21:-17], axis=1)
    X_sprout = dna_to_onehot(sprout_test_df['target_seq'])
    y_sprout = sprout_test_df['prob_1bpins']

    regressor = joblib.load(model_save_path)
    y_pred = regressor.predict(X_test)
    lindel_pred = regressor.predict(X_lindel)
    sprout_pred = regressor.predict(X_sprout)
    sprout_rp, _ = pearsonr(y_sprout, sprout_pred)
    cor_list.append(sprout_rp)
    k_t, _ = kendalltau(y_sprout, sprout_pred)
    k_list.append(k_t)

    forecast_eval_result = get_eval(y_test, y_pred)
    lindel_eval_result = get_eval(y_lindel, lindel_pred)
    print(f'forecast mse: {forecast_eval_result[0]}, lindel mse: {lindel_eval_result[0]}')
    print(f'forecast pearson: {forecast_eval_result[1]}, lindel pearson: {lindel_eval_result[1]}')
    print(f'random state: {seed}, sprout pearson cor: {sprout_rp}, k_t: {k_t}')

model = XGBRegressor(objective='reg:logistic', reg_alpha=0.2, reg_lambda=0.8)
train_valid(forecast_lindel_train_path, forecast_lindel_valid_path, model, model_save_path)
test([test_path, lindel_test_path, sprout_test_path], model_save_path, seed=300)