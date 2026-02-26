from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models_config():
    return {
        "Logistic": (LogisticRegression(max_iter=1000), "linear"),
        "RandomForest": (RandomForestClassifier(), "tree"),
        "XGB": (XGBClassifier(eval_metric="logloss"), "tree")
    }

def get_param_grids():
  return {
    "Logistic": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__penalty": ["l2"]
    },
    "RandomForest": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [5, 10,15],
        "clf__min_samples_split": [2, 5, 10]
    },
    "XGB": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.01, 0.1, 0.2],
        "clf__max_depth": [3, 5, 7],
        "clf__subsample": [0.8, 1.0]
    }
}
