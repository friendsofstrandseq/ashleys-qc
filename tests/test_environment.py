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
    err_msg = ''
    try:
        import pysam
    except ImportError:
        err_msg += 'Module pysam could not be imported.\n'
    
    try:
        import scipy
    except ImportError:
        err_msg += 'Module "scipy" could not be imported.\n'

    try:
        import matplotlib
    except ImportError:
        err_msg += 'Module "matplotlib" could not be imported.\n'

    try:
        import numpy
    except ImportError:
        err_msg += 'Module "numpy" could not be imported.\n'

    if err_msg:
        raise RuntimeError('ERROR: at least one external module import failed; '
                            'please make sure that your environment setup is complete. '
                            'Recorded import errors:\n{}'.format(err_msg))

    return True
