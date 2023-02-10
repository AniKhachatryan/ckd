"""
test_evaluate.py
================

Test the evaluate.py module.
"""


from ckd.evaluate import evaluate
from math import isclose
import numpy as np


def test_evaluate():
    y_true = ['ckd', 'ckd', 'ckd', 'notckd', 'notckd', 'notckd']
    y_predicted = ['ckd', 'ckd', 'notckd', 'notckd', 'notckd', 'ckd']

    performance_metrics = evaluate(y_true, y_predicted, verbose=False)

    assert isclose(performance_metrics['accuracy'], 0.66, abs_tol=.01)
    assert isclose(performance_metrics['recall_sensitivity'], 0.66, abs_tol=.01)
    assert isclose(performance_metrics['specificity'], 0.66, abs_tol=.01)
    assert isclose(performance_metrics['precision'], 0.66, abs_tol=.01)
    assert np.array_equal(performance_metrics['confusion_mat'], np.asarray([[2, 1], [1, 2]]))
