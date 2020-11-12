from aroma.tests.conftest import features_df
import numpy as np
import os.path as op
import pandas as pd

import pytest

from aroma import features
from aroma.tests.utils import get_tests_resource_path


def test_feature_time_series(mel_mix, mc, features_df):

    # Run feature_time_series
    max_RP_corr = features.feature_time_series(mel_mix, mc)

    # Read features csv
    df = pd.read_csv(features_df)

    assert np.allclose(df["max_RP_corr"].values[:5], max_RP_corr[:5], atol=1e-1)


def test_feature_frequency(mel_FT_mix, features_df):

    # Run feature_frequency
    HFC = features.feature_frequency(mel_FT_mix, TR=2)

    # Read features csv
    df = pd.read_csv(features_df)

    assert np.allclose(df["HFC"].values, HFC, atol=1e-2)


def test_feature_spatial(mel_IC, features_df):

    # Run feature_spatial
    edge_fract, csf_fract = features.feature_spatial(mel_IC)

    # Read features csv
    df = pd.read_csv(features_df)

    assert np.allclose(df["edge_fract"].values, edge_fract, atol=1e-2)
    assert np.allclose(df["csf_fract"].values, csf_fract, atol=1e-2)
