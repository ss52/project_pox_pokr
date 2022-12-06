import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
from pathlib import Path
import wandb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from wandb.lightgbm import wandb_callback, log_summary


load_dotenv(override=True)

WORK_DIR = os.getenv("DATA_DIR")
FILE_NAME = 'df_work_test.zip'
filepath = Path(WORK_DIR, FILE_NAME)

data = pd.read_csv(filepath, index_col=0, parse_dates=True)

X = data.drop(['Fe2+', 'ac'], axis=1)
y = data['Fe2+']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


def train_evaluate():
    wandb.init()

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'max_depth': wandb.config.max_depth,
        'num_leaves': wandb.config.num_leaves,
        # 'feature_fraction': 0.8,
        # 'subsample': 0.2,
        'max_bin': wandb.config.max_bin,
        'lambda_l1': wandb.config.lambda_l1,
        'lambda_l2': wandb.config.lambda_l2
    }

    model = lgb.train(params, train_data,
                      num_boost_round=500,
                      early_stopping_rounds=50,
                      valid_sets=[valid_data],
                      valid_names=['valid'])

    y_pred = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, y_pred)
    rmse = mean_squared_error(y_valid, y_pred) ** 0.5

    wandb.log({"mae": mae, "rmse": rmse})
    log_summary(model)


sweep_configs = {
    "method": "grid",
    'name': 'lgbm',
    "metric": {"name": "rmse", "goal": "minimize"},
    "parameters": {
        'learning_rate': {"values": [0.01, 0.05, 0.1, 0.2]},
        'max_depth': {'values': [-1, 10, 20]},
        'num_leaves': {"values": [31, 100, 150, 200]},
        # 'feature_fraction': 0.8,
        # 'subsample': 0.2,
        'max_bin': {"values": [100, 150, 200]},
        'lambda_l1': {"value": 0},
        'lambda_l2': {"value": 0}
    }
}


sweep_id = wandb.sweep(sweep_configs, project="PAGK")
wandb.agent(sweep_id=sweep_id, function=train_evaluate)
