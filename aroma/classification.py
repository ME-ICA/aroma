# CHANGES
# -------
# Log of changes as mandated by the original Apache 2.0 License of ICA-AROMA
#
#   * Drop ``runICA`` and ``register2MNI`` functions
#   * Base ``classifier`` on Pandas, and revise signature (``predict(X)``)
#     to make it more similar to scikit learn
#   * Return classification labels directly on ``predict``
#
"""Classification functions for ICA-AROMA."""
import logging

import numpy as np

LGR = logging.getLogger(__name__)

# Define criteria needed for classification (thresholds and
# hyperplane-parameters)
THR_CSF = 0.10
THR_HFC = 0.35
HYPERPLANE = [-19.9751070082159, 9.95127547670627, 24.8333160239175]


def predict(X, thr_csf=THR_CSF, thr_hfc=THR_HFC, hplane=HYPERPLANE):
    """
    Classify components as motion or non-motion based on four features.

    The four features used for classification are: maximum RP correlation,
    high-frequency content, edge-fraction, and CSF-fraction.

    Parameters
    ----------
    X : :obj:`pandas.DataFrame`
        Features table (C x 4), must contain the following columns:
        "edge_fract", "csf_fract", "max_RP_corr", and "HFC".

    Returns
    -------
    y : array_like
        Classification (``True`` if the component is a motion one).

    """
    # Project edge & max_RP_corr feature scores to new 1D space
    x = X[["max_RP_corr", "edge_fract"]].values
    proj = (hplane[0] + np.dot(x.T, hplane[1:])) > 0

    # Classify the ICs
    return (X["csf_fract"] > thr_csf) | (X["HFC"] > thr_hfc) | proj
