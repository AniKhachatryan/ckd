"""
test_preprocess.py
==================

Test the preprocess.py module.
"""


import pandas as pd
from ckd.preprocess import preprocess_data
from ckd.config import Config
from ckd.data import load_test_data, check_data


def test_preprocess_data():
    config = Config()
    x, y = load_test_data(config, preprocessed=False)

    x_preprocessed = preprocess_data(x, config)

    assert isinstance(x_preprocessed, pd.DataFrame)
    assert x_preprocessed.shape[1] == 35
    assert check_data(x_preprocessed, config, preprocessed=True)
