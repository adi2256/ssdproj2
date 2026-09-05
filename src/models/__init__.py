"""
Prediction models.

All expose the same interface:

    fit(X, y)               -> self
    predict_proba(X) -> p    (n,) array of P(fails within horizon)

`p` is 1-D, not sklearn's (n, 2). The conformal layer takes P(y=1).

    random_forest   the base paper's back end; takes FLATTENED windows
    gru             temporal model; takes 3-D windows

MODELS maps a name to a factory taking (seed). MODEL_INPUT says which
shape each expects, so the LOMO runner can flatten only when needed.
"""

from .baseline import RandomForestBaseline, make_baseline
from .sequence import HAVE_TORCH, SequenceModel, make_sequence_model

MODELS = {
    "random_forest": make_baseline,
    "gru": make_sequence_model,
}

# "flat" -> (n, features); "seq" -> (n, window_len, features)
MODEL_INPUT = {
    "random_forest": "flat",
    "gru": "seq",
}

__all__ = ["MODELS", "MODEL_INPUT", "HAVE_TORCH",
           "RandomForestBaseline", "SequenceModel",
           "make_baseline", "make_sequence_model"]
