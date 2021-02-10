import pickle
import warnings
import pandas as pd
import pytest
import sys
import sklearn

from ashleyslib.prediction import predict_model


def test_pickle_warning():
    model_name = 'models/svc_default.pkl'
    features = pd.read_csv('data/test_features.tsv', sep='\t')
    features = features.drop(columns=['sample_name'])

    if sklearn.__version__ == '0.23.2':
        with pytest.warns(None) as record:
            predict_model(model_name, features)
        assert len(record) == 0
    else:
        with pytest.warns(UserWarning, match='You are using a different version of scikit-learn') as record:
            predict_model(model_name, features)
        assert len(record) == 1

    return True


def test_module_import():
    import pysam
    import matplotlib
    import numpy
    assert 'pysam' in sys.modules
    assert 'matplotlib' in sys.modules
    assert 'numpy' in sys.modules
    return True
