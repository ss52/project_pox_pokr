import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
from pathlib import Path
from wandb.lightgbm import wandb_callback, log_summary
import wandb

load_dotenv(override=True)

wandb.init(project="PAGK", entity="smarkelov")

WORK_DIR = os.getenv("DATA_DIR")
FILE_NAME = 'df_work_test.zip'
filepath = Path(WORK_DIR, FILE_NAME)

params = {
                 'objective': 'regression',
                 'metric': 'rmse',
                'learning_rate': 0.05,
                 'max_depth': -1,
                 'num_leaves': 200,
                 # 'feature_fraction': 0.8,
                 # 'subsample': 0.2,
                 'max_bin': 500,
                 'lambda_l1': 0,
                 'lambda_l2': 0,
                 }

wandb.config.update(params)


def train_evaluate(params):

    data = pd.read_csv(filepath, index_col=0, parse_dates=True)

    X = data.drop(['Fe2+', 'ac'], axis=1)
    y = data['Fe2+']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    model = lgb.train(params, train_data,
                      num_boost_round=1000,
                      early_stopping_rounds=100,
                      valid_sets=[valid_data],
                      valid_names=['valid'],
                      callbacks=[wandb_callback()])

    score = model.best_score['valid']['rmse']

    log_summary(model)

    return score


if __name__ == '__main__':
    score = train_evaluate(params)
    # wandb.log({"RMSE": score})
    print('validation RMSE:', score)
