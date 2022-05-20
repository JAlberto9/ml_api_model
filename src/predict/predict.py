import imp
import pandas as pd
import xgboost as xgb
from model import load_model

try:
    model = load_model()
    print('Model load XD')
except KeyError:
    print('KeyError: missing model')

path_csv = 'src/data/orders_test.csv'
path_csv_score = 'src/data/orders_test_scored.csv'

FEATURES = [
    'to_user_distance',
    'to_user_elevation',
    'total_earning',
]


def score(df, model):
    data_m = xgb.DMatrix(df[FEATURES])
    df['taken'] = model.predict(data_m)
    df['taken_score'] = df['taken']
    df['taken'] = (df['taken'] > .5).astype(int)
    return df


def predict():
    df = pd.read_csv(path_csv)
    print(df.head)
    data = score(df, model)
    print(data.head())
    data.to_csv(path_csv_score)


if __name__ == '__main__':
    predict()
