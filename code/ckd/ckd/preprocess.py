"""
"""

import pickle
import pandas as pd
from ckd.data import check_data


def preprocess_data(df, config, test=True):

    # assert column names
    assert check_data(df, config, preprocessed=False)

    # missing values
    missing_value_imputer = pickle.load(open(config.missing_value_imputer_path, 'rb'))
    column_names_new = [column_name[5:] for column_name in missing_value_imputer.get_feature_names_out()]
    if test:
        # transform
        df = pd.DataFrame(missing_value_imputer.transform(df), index=df.index, columns=column_names_new)
    else:
        # fit transform
        df = pd.DataFrame(missing_value_imputer.fit_transform(df), index=df.index, columns=column_names_new)

    # encode categorical columns
    df = pd.get_dummies(df, columns=config.column_names_cat, drop_first=True)

    # scale features
    robust_scaler = pickle.load(open(config.robust_scaler_path, 'rb'))
    if test:
        # transform
        df[config.column_names_cat] = robust_scaler.transform(df[config.column_names_cat])
    else:
        # fit transform
        df[config.column_names_cat] = robust_scaler.fit_transform(df[config.column_names_cat])

    return df