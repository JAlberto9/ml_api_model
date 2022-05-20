import xgboost as xgb
import numpy as np


def load_model():
    model_name = 'src/model/MODEL_CLASSIFIEr'
    bst = xgb.Booster()
    bst.load_model(fname=model_name + ".xgb")
    return bst
