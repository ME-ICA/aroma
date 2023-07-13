"""Tests for aroma.classification."""
import pandas as pd
from aroma import classification


def test_classification(classification_overview):
    """Test aroma.utils.classification and ensure classifications come out the same."""
    clf_overview_df = pd.read_table(classification_overview)
    test_df = clf_overview_df[["edge_fract", "csf_fract", "max_RP_corr", "HFC"]]
    test_classifications = classification.predict(test_df, metric_metadata={})
    true_classifications = clf_overview_df["classification"].tolist()
    assert true_classifications == test_classifications
