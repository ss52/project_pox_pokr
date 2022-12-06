import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
import os
from pathlib import Path
from clearml import Task

load_dotenv(override=True)

task = Task.init(project_name='PAGK', task_name='lightGBM search params 5')

WORK_DIR = os.getenv("DATA_DIR")
FILE_NAME = 'df_work_test.zip'
filepath = Path(WORK_DIR, FILE_NAME)

SEARCH_PARAMS = {'learning_rate': 0.05,
                 'max_depth': -1,
                 'num_leaves': 31,
                 # 'feature_fraction': 0.8,
                 # 'subsample': 0.2,
                 'max_bin': 255,
                 'lambda_l1': 0,
                 'lambda_l2': 0,
                 }
task.connect(SEARCH_PARAMS)


def train_evaluate(search_params):

    data = pd.read_csv(filepath, index_col=0, parse_dates=True)

    X = data.drop(['Fe2+', 'ac'], axis=1)
    y = data['Fe2+']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    params = {'objective': 'regression',
              'metric': 'rmse',
              **search_params}

    model = lgb.train(params, train_data,
                      num_boost_round=3000,
                      early_stopping_rounds=300,
                      valid_sets=[valid_data],
                      valid_names=['valid'])

    score = model.best_score['valid']['rmse']
    return score


if __name__ == '__main__':
    score = train_evaluate(SEARCH_PARAMS)
    print('validation RMSE:', score)
