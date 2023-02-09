"""
"""


from ckd.config import Config
from ckd.data import load_data, check_data
from ckd.preprocess import preprocess_data
from ckd.predict_utils import load_model
from ckd.evaluate import evaluate


def predict_ckd(input_data, model='LR', preprocess=True,
                evaluate_predictions=True, config=Config()):

    # input_data is either 'default', a path, a pandas.DataFrame
    x, y = load_data(input_data, config)

    # check that data has the right format (columns)
    assert check_data(x, config, preprocessed=not preprocess)

    if preprocess:
        x = preprocess_data(x)

    # handle model
    model = load_model(model, config)

    # predict
    y_predicted = model.predict(x)

    evaluation_dict = None
    if evaluate_predictions:
        # TODO
        # make sure we have valid y_true
        assert y is not None
        evaluation_dict = evaluate(y, y_predicted)

    # return predictions and performance metrics dict
    return y_predicted, evaluation_dict
