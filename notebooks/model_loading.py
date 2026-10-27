"""Load the saved final models on any operating system.

The models in outputs/models/ were saved on macOS/Linux. A pickled XGBoost model also stores the
state of its random number generator (`rng_state`), and the Windows build of XGBoost writes that
state in a different text format, so on Windows `joblib.load` fails with "input stream corrupted".

Prediction never uses the random number generator. So if the normal load fails, we load only the
trees and the objective (the "Model" part of the saved buffer) and skip the training settings.
The scores are the same as on macOS/Linux.
"""
import joblib
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
