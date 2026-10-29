"""Load the saved final models on any operating system, and turn their scores into probabilities.

load_model: the models in outputs/models/ were saved on macOS/Linux. A pickled XGBoost model also stores the
state of its random number generator (`rng_state`), and the Windows build of XGBoost writes that
state in a different text format, so on Windows `joblib.load` fails with "input stream corrupted".

Prediction never uses the random number generator. So if the normal load fails, we load only the
trees and the objective (the "Model" part of the saved buffer) and skip the training settings.
The scores are the same as on macOS/Linux.

calibrate: the scores are not probabilities (fraud was weighted about 26x in training). Notebook 04
(section 7.4) fits a Platt calibrator on Base month 5 and saves it to configs/calibration.json.
"""
import json

import joblib
import numpy as np
from xgboost import Booster, core
from xgboost.core import XGBoostError

# In XGBoost's binary format the buffer is {"Config": {...}, "Model": {...}}; this is the "Model" key
_MODEL_KEY = b'L\x00\x00\x00\x00\x00\x00\x00\x05Model'


def _load_trees_only(self, state):
    raw = bytes(state['handle'])
    start = raw.index(_MODEL_KEY) + len(_MODEL_KEY)
    Booster.__init__(self)
    self.load_model(bytearray(raw[start:-1]))  # drop the outer closing brace


def load_model(path):
    try:
        return joblib.load(path)
    except XGBoostError:
        normal_setstate = core.Booster.__setstate__
        core.Booster.__setstate__ = _load_trees_only
        try:
            return joblib.load(path)
        finally:
            core.Booster.__setstate__ = normal_setstate


def calibrate(scores, seed, fraud_rate=None, path='../configs/calibration.json'):
    """Probability of fraud for the raw scores of the given seed's model.

    fraud_rate: if the fraud rate is known to differ from the month the calibrator was fitted on,
    the probabilities are corrected to it (prior shift) without refitting.
    """
    calibration = json.load(open(path))
    a, b = calibration['seeds'][str(seed)]['a'], calibration['seeds'][str(seed)]['b']
    clip = calibration['score_clip']
    s = np.clip(np.asarray(scores, dtype=float), clip, 1 - clip)
    log_odds = a * np.log(s / (1 - s)) + b
    if fraud_rate is not None:
        fitted_rate = calibration['fraud_rate_when_fitted']
        log_odds += np.log(fraud_rate / (1 - fraud_rate)) - np.log(fitted_rate / (1 - fitted_rate))
    return 1 / (1 + np.exp(-log_odds))
