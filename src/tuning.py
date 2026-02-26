from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

def get_pipeline(model, strategy, cat_cols, num_cols, model_type="linear", pos_weight=1):
    model = clone(model)

    if strategy == "Class_Weights":
        if hasattr(model, "class_weight"):
            model.set_params(class_weight="balanced")
        elif isinstance(model, XGBClassifier):
            model.set_params(scale_pos_weight=pos_weight)

    if model_type == "linear":
        num_transformer = StandardScaler()
    else:
        num_transformer = "passthrough"

    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    steps = [("pre", preprocessor)]
    if strategy == "SMOTE":
        steps.append(("smote", SMOTE(random_state=42)))

    steps.append(("clf", model))
    return ImbPipeline(steps)
