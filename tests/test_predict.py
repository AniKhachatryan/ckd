"""
test_predict.py
===============

Test the predict.py module.
"""


import numpy as np
from ckd.predict import predict_ckd


def test_predict_ckd():
    y_predicted, evaluation_dict = predict_ckd()

    assert isinstance(y_predicted, np.ndarray)
    assert isinstance(evaluation_dict, dict)
