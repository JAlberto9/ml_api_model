# from conecctions.conecctions import save_p
from asyncore import read
from xml.sax.handler import feature_external_ges
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import matplotlib.pyplot as plt
from data import FEATURES
# import seaborn as sns

import xgboost as xgb
import shap
import os
import joblib
import json

# Options
# version = 'v1'
repo = os.path.abspath(os.getcwd())
metric_name = 'taken'
test_size = 0.20
random_state = 1337

# A parameter grid for XGBoost
params = {
    'min_child_weight': [1, 3, 5],
    'gamma': [0.5],
    'subsample': [0.5],
    'colsample_bytree': [0.5],
    'colsample_bylevel': [0.5],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.1, 0.01],
    'max_delta_step': [0.5],
    'n_estimators': [10, 50, 100, 200],
    'reg_alpha': [0.5],
    'reg_lambda': [0.5]
}


def eval_metrics(actual, pred):
    auc = metrics.roc_auc_score(actual, pred)
    return auc


def save_model(model_name, bst_model, features, folder_name):
    name = folder_name + "/" + model_name
    bst_model.save_model(fname=name + ".xgb")
    np.save(file=name + ".npy", arr=np.array(features))


def train():
    # mlflow.xgboost.autolog()
    train_dataset_path = 'src/data/datase.csv'
    model_path_xgb = 'src/model/MODEL_CLASSIFIEr'
    scores_file = 'src/metrics/scores.json'

    # Initialize XGB model∫
    clf = xgb.XGBClassifier(
        # objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        seed=42
    )

    # GridSearch
    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=params,
        scoring='roc_auc',
        cv=5
    )

    # Read Train/Test
    dataset = pd.read_csv(train_dataset_path)
    print(dataset.columns)
    X = dataset[FEATURES]
    y = dataset[metric_name]
    print('Splitting...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state)

    # Train
    print('Training...')
    models = gsearch.fit(X_train, y_train)
    print(models.best_score_)

    model = models.best_estimator_
    # Check the accuracy
    print("Train score: " + str(model.score(X_train, y_train)))
    print("Test score: " + str(model.score(X_test, y_test)))

    # Feature importance
    top_features = pd.DataFrame()
    top_features['columns'] = X_train.columns
    top_features['importances'] = model.feature_importances_
    top_features = top_features.set_index('columns')
    top_features.sort_values(by='importances', ascending=False, inplace=True)
    top_features.to_csv('src/data/top_features_importance.csv')
    top_features = top_features[:50]
    print(top_features)
    print("")
    # Save model
    print('Exporting model...')
    model.save_model(fname=model_path_xgb + ".xgb")

    # Confusion Matrix Train
    y_pred = model.predict_proba(X_train)[:, 1]
    # y_pred.to_csv(y_test, index=False)
    # pd.cut(y_pred, 10).value_counts().plot(kind='bar')
    # plt.show()

    precision, recall, thresholds = metrics.precision_recall_curve(
        y_train, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision + 0.001)
    threshold = float(thresholds[np.argmax(f1_scores)])
    print("Threshold: %s" % threshold)
    roc_auc = eval_metrics(y_train, y_pred)
    print("AUC: %s" % roc_auc)
    y_pred = (y_pred >= threshold).astype(int)
    print(metrics.confusion_matrix(y_train, y_pred))

    # Confusion Matrix Test
    y_pred = model.predict_proba(X_test)[:, 1]
    # sns.distplot(y_pred)
    #pd.cut(y_pred, 10).value_counts().plot(kind='bar')
    # plt.show()
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_test, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision + 0.001)
    threshold = float(thresholds[np.argmax(f1_scores)])
    print("Threshold: %s" % threshold)
    roc_auc = eval_metrics(y_test, y_pred)
    print("AUC: %s" % roc_auc)

    # Save classes
    classes = pd.DataFrame(y_pred).join(y_test)
    classes.columns = ['predicted', 'actual']
    classes.to_csv('src/data/classes.csv', index=False)

    y_pred = (y_pred >= threshold).astype(int)
    print(metrics.confusion_matrix(y_test, y_pred))

    print(metrics.classification_report(y_test, y_pred))

    with open(scores_file, 'w') as fd:
        json.dump({'roc_auc': roc_auc, 'threshold': threshold}, fd, indent=4)

    # Explainability
    print('Explainability...')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test[X_train.columns])

    # Summarize the effects of all the features
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    # plt.show()


if __name__ == '__main__':
    train()
