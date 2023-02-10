"""
test_config.py
==============

Test the config.py module.
"""


from ckd.config import Config
from ckd.utils import get_root
import os


def test_config():
    config = Config()

    assert os.path.isfile(os.path.join(get_root(), config.x_train_path))
    assert os.path.isfile(os.path.join(get_root(), config.x_test_path))
    assert os.path.isfile(os.path.join(get_root(), config.x_train_preprocessed_path))
    assert os.path.isfile(os.path.join(get_root(), config.x_test_preprocessed_path))
    assert os.path.isfile(os.path.join(get_root(), config.y_train_path))
    assert os.path.isfile(os.path.join(get_root(), config.y_test_path))
    assert isinstance(config.target_column_name, str)
    assert os.path.isfile(os.path.join(get_root(), config.model_path_lr))
    assert os.path.isfile(os.path.join(get_root(), config.model_path_rf))
    assert os.path.isfile(os.path.join(get_root(), config.missing_value_imputer_path))
    assert os.path.isfile(os.path.join(get_root(), config.robust_scaler_path))
    assert isinstance(config.column_names, tuple)
    assert isinstance(config.column_names_preprocessed, tuple)
    assert isinstance(config.column_names_cat, tuple)
    assert isinstance(config.column_names_num, tuple)