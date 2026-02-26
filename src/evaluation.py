import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from scipy.stats import ttest_rel, wilcoxon

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def cohens_d(x, y):
    diff = x - y
    return diff.mean() / (diff.std(ddof=1) + 1e-9)

def statistical_comparison(results_df, dataset_name, metric="roc_auc"):
    dataset_df = results_df[results_df["dataset"] == dataset_name]
    comparisons = []
    strategies = dataset_df["strategy"].unique()

    for strategy in strategies:
        strat_df = dataset_df[dataset_df["strategy"] == strategy]
        models = strat_df["model"].unique()

        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model_a, model_b = models[i], models[j]

                scores_a = strat_df[strat_df["model"] == model_a][metric].values
                scores_b = strat_df[strat_df["model"] == model_b][metric].values

                t_stat, t_p = ttest_rel(scores_a, scores_b)
                try:
                    _, w_p = wilcoxon(scores_a, scores_b)
                except:
                    w_p = np.nan

                comparisons.append({
                    "dataset": dataset_name,
                    "strategy": strategy,
                    "model_A": model_a,
                    "model_B": model_b,
                    "metric": metric,
                    "mean_A": scores_a.mean(),
                    "mean_B": scores_b.mean(),
                    "t_p_value": t_p,
                    "wilcoxon_p_value": w_p,
                    "cohens_d": cohens_d(scores_a, scores_b)
                })

    return pd.DataFrame(comparisons)
