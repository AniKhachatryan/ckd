"""

"""


import os
import pickle
import sklearn


def load_model(model, config):

    # check if model is already a sklearn model
    if isinstance(model, sklearn.base.BaseEstimator):
        return model

    # otherwise assert that model is a string
    assert isinstance(model, str)

    if model.lower() == 'lr':
        model_loaded = pickle.load(open(config.model_path_lr, 'rb'))
    elif model.lower() in ['rf', 'rfc']:
        model_loaded = pickle.load(open(config.model_path_rf, 'rb'))
    elif os.path.isfile(model):
        # assert that the path is a pickle file a pickle file
        assert os.path.splitext(model)[1] == '.pkl'
        # load the model
        model_loaded = pickle.load(open(model, 'rb'))
        # assert that it's a sklearn.base.BaseEstimator
        assert isinstance(model, sklearn.base.BaseEstimator)
    else:
        raise ValueError('Invalid model argument passed.')

    return model_loaded
