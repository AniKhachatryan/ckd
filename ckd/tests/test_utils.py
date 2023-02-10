"""
test_utils.py
=============

Test the utils.py module.
"""


from ckd.utils import load_model, get_root
from ckd.config import Config
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def test_load_model():
    config = Config()
    print(type(load_model('lr', config)))
    assert isinstance(load_model('lr', config), LogisticRegression), 'lr'
    assert isinstance(load_model('rf', config), RandomForestClassifier), 'rf'


def test_get_root():
    assert os.path.exists(get_root())