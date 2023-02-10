"""
test_data.py
============

Test the data.py module.
"""

from ckd.data import load_test_data, load_data, check_data
from ckd.config import Config
import pandas as pd


def test_load_test_data():
    config = Config()

    # preprocessed
    x, y = load_test_data(config, preprocessed=True)
    assert isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame)
    assert x.shape[1] == 35

    # raw
    x_raw, y_raw = load_test_data(config, preprocessed=False)
    assert isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame)
    assert x_raw.shape[1] == 24


def test_load_train_data():
    pass


def test_load_data():
    config = Config()

    x, y = load_data('default', config)
    assert isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame)
    assert x.shape[1] == 35
