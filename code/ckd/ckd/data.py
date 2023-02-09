"""

"""


import os.path
import pandas
import pandas as pd


def load_test_data(config, preprocessed=True):

    # load x
    if preprocessed:
        x = pd.read_csv(config.x_test_preprocessed_path, index_col=0)
    else:
        x = pd.read_csv(config.x_test_path, index_col=0)

    # load y
    y = pd.read_csv(config.y_test_path, index_col=0)

    return x, y


def load_train_data(config, preprocessed=True):
    # To be developed later when I decide to add training functionality
    raise NotImplementedError


def load_data(input_data, config):

    # if 'default', load default data
    if input_data == 'default':
        x, y = load_test_data(config)
        return x, y

    # if pandas.DataFrame
    if isinstance(input_data, pandas.DataFrame):
        pass
    # else, make sure it's a path and load it as a pandas.DataFrame
    elif os.path.isfile(input_data):
        # make sure it's a .csv file
        assert os.path.splitext(input_data)[1] == '.csv'
        # load the data, assume first column is the index
        input_data = pd.read_csv(input_data, index_col=0)
    else:
        raise ValueError('Invalid input_data.')

    # check if input_data contains the target variable
    if config.target_colname in input_data.columns:
        x = input_data.drop(columns=config.target_colname)
        y = input_data[config.target_colname]
    else:
        x = input_data
        y = None

    return x, y


def check_data(df, config, preprocessed=True):
    if preprocessed:
        column_names = config.colnames_preprocessed
    else:
        column_names = config.colnames

    # if something is wrong raise an error with a helpful error message
    assert set(df.columns) == set(column_names), 'Incorrect column names.'

    return True
