"""Tests for the features module."""
import numpy as np
from aroma import features


def test_feature_time_series(mel_mix, mc, max_correls):
    """Test the feature_time_series feature against pre-calculated values."""
    np.random.seed(1)

    # Run feature_time_series
    max_RP_corr = features.feature_time_series(mel_mix, mc)

    # Read features csv
    max_correls = np.load(max_correls)

    assert np.allclose(max_correls, max_RP_corr, atol=1e-2)


def test_feature_frequency(mel_FT_mix, HFC):
    """Test the feature_frequency feature against pre-calculated values."""
    np.random.seed(1)

    # Run feature_frequency
    HFC_test = features.feature_frequency(mel_FT_mix, TR=2)

    # Read features csv
    HFC = np.load(HFC)

    assert np.allclose(HFC, HFC_test)


def test_feature_spatial(mel_IC, edgeFract, csfFract):
    """Test the feature_spatial features against pre-calculated values."""
    np.random.seed(1)

    # Run feature_spatial
    edge_fract, csf_fract = features.feature_spatial(mel_IC)

    # Read features csv
    edgeFract = np.load(edgeFract)
    csfFract = np.load(csfFract)

    assert np.allclose(edgeFract, edge_fract)
    assert np.allclose(csfFract, csf_fract)
