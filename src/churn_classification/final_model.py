"""Final chosen model for the churn classification task.

Selected in notebooks/churn_classification/03_modeling.ipynb after 3 rounds of
comparison (untuned baseline, manual regularization, Optuna TPE search over
25 trials/model across Logistic Regression + LightGBM + XGBoost + CatBoost,
all under the same StratifiedKFold(5) CV): CatBoost won on PR-AUC (0.6768,
5-fold CV mean on the train pool) with a healthy overfit_gap (0.0257). See
docs/docs.md Pha 3 for the full comparison table and reasoning.

BEST_PARAMS is the literal output of that Optuna search — not re-tuned here.
"""

from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

from src.churn_classification.preprocessing import build_preprocessor

BEST_PARAMS = dict(
    max_depth=4,
    n_estimators=239,
    learning_rate=0.02667319732421495,
    l2_leaf_reg=1.0462122627760353,
    min_data_in_leaf=35,
    subsample=0.8684947869402209,
)


def build_final_pipeline(scale_pos_weight: float, random_state: int = 42) -> Pipeline:
    """Preprocessing + tuned CatBoost, ready to .fit() on a train set.

    scale_pos_weight is taken as a parameter (not hardcoded) because it must
    always be recomputed from whatever training data is actually being fit
    on (see compute_scale_pos_weight) rather than frozen at search time.
    """
    model = CatBoostClassifier(
        **BEST_PARAMS,
        bootstrap_type="Bernoulli",
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbose=False,
    )
    return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
