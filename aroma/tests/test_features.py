"""Tests for the features module."""
import numpy as np
from aroma import features


def test_feature_time_series(mel_mix, mc, max_correls):
    """Test the feature_time_series feature against pre-calculated values."""
    np.random.seed(1)

    # Read mel_mix
    mel_mix = np.loadtxt(mel_mix)

    # Run feature_time_series
    max_RP_corr, _ = features.feature_time_series(mel_mix, mc)

    # Read features csv
    max_correls = np.load(max_correls)

    assert np.allclose(max_correls, max_RP_corr, atol=1e-2)

    # Run feature_time_series with metric metadata
    metadata = {}
    max_RP_corr, updated_metadata = features.feature_time_series(
        mel_mix,
        mc,
        metric_metadata=metadata,
    )
    assert "max_RP_corr" in updated_metadata.keys()


def test_feature_frequency(mel_FT_mix, HFC):
    """Test the feature_frequency feature against pre-calculated values."""
    np.random.seed(1)

    # Read mel_FT_mix
    mel_FT_mix = np.loadtxt(mel_FT_mix)

    # Run feature_frequency
    HFC_test, _ = features.feature_frequency(mel_FT_mix, TR=2)

    # Read features csv
    HFC = np.load(HFC)

    assert np.allclose(HFC, HFC_test)

    # Run feature_frequency with metric metadata
    metadata = {}
    HFC_test, updated_metadata = features.feature_frequency(
        mel_FT_mix,
        TR=2,
        metric_metadata=metadata,
    )
    assert "HFC" in updated_metadata.keys()


def test_feature_spatial(mel_IC, edgeFract, csfFract):
    """Test the feature_spatial features against pre-calculated values."""
    np.random.seed(1)

    # Run feature_spatial
    edge_fract, csf_fract, _ = features.feature_spatial(mel_IC)

    # Read features csv
    edgeFract = np.load(edgeFract)
    csfFract = np.load(csfFract)

    assert np.allclose(edgeFract, edge_fract)
    assert np.allclose(csfFract, csf_fract)

    # Run feature_spatial with metric metadata
    metadata = {}
    edge_fract, csf_fract, updated_metadata = features.feature_spatial(mel_IC, metadata)
    assert "edge_fract" in updated_metadata.keys()
    assert "csf_fract" in updated_metadata.keys()
