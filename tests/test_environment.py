import pickle
import warnings
import pandas as pd
import pytest

from ashleyslib.prediction import predict_model


def test_pickle_warning():
    model_name = 'models/svc_default.pkl'
    features = pd.read_csv('data/test_features.tsv', sep='\t')
    features = features.drop(columns=['sample_name'])

    with pytest.warns(None) as record:
        predict_model(model_name, features)
    assert len(record) == 0

    return True
