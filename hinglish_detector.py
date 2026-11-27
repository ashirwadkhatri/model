"""Runtime Hinglish detector. Loads hinglish_model.joblib (made by train_hinglish.py).

    from hinglish_detector import is_hinglish, hinglish_prob
    is_hinglish("ye movie bahut mast hai")   # True
If the model file is missing, everything returns False/0.0 and only the word list in langfilter.py is used.
"""
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("HINGLISH_MODEL", os.path.join(HERE, "hinglish_model.joblib"))
THRESHOLD = float(os.environ.get("HINGLISH_THRESHOLD", "0.8"))   # overridden by the threshold saved in the model, unless this env var is set
_ENV_T = "HINGLISH_THRESHOLD" in os.environ
_model = None
_loaded = False


def squeeze(t):                                   # "karooooo" -> "karo"  (module-level so joblib can pickle it)
    return re.sub(r"(.)\1{2,}", r"\1", str(t).lower())


def _get():
    global _model, _loaded
    if not _loaded:
        _loaded = True
        if os.path.exists(MODEL_PATH):
            import joblib
            _model = joblib.load(MODEL_PATH)
            if isinstance(_model, dict):                      # new format: {"model":..., "threshold":...}
                global THRESHOLD
                if not _ENV_T:
                    THRESHOLD = float(_model["threshold"])
                _model = _model["model"]
        else:
            print(f"[hinglish_detector] {MODEL_PATH} not found - using word list only", flush=True)
    return _model


def available():
    return _get() is not None


def threshold():
    _get()
    return THRESHOLD


def hinglish_probs(texts):
    m = _get()
    if m is None or not texts:
        return [0.0] * len(texts)
    return [float(p) for p in m.predict_proba(texts)[:, 1]]


def hinglish_prob(text):
    return hinglish_probs([text])[0]


def is_hinglish(text):
    return hinglish_prob(text) >= THRESHOLD


def flags(texts):
    """True for each text the detector says is Hinglish (uses the threshold saved in the model)."""
    return [p >= THRESHOLD for p in hinglish_probs(texts)]
