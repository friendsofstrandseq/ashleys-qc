import pickle
import warnings
import pandas as pd

from ashleyslib.prediction import predict_model
warnings_as_errors=all


def test_pickle_warning(capsys):
    model_name = 'models/svc_default.pkl'
    features = pd.read_csv('data/test_features.tsv', sep='\t')
    features = features.drop(columns=['sample_name'])

    predict_model(model_name, features)
    captured = capsys.readouterr()
    assert not captured.err

    return True
