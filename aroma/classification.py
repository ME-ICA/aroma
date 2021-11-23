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
HYPERPLANE = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])


def hfc_criteria(x, thr_hfc=THR_HFC):
    """
    Compute the HFC criteria for classification.

    Parameters
    ----------
    x : numpy.ndarray
        Projection of HFC feature scores to new 1D space.

    Returns
    -------
    numpy.ndarray
        Classification (``True`` if the component is a motion one).
    """
    return x > thr_hfc


def csf_criteria(x, thr_csf=THR_CSF):
    """
    Compute the CSF criteria for classification.

    Parameters
    ----------
    x : numpy.ndarray
        Projection of CSF-fraction feature scores to new 1D space.

    Returns
    -------
    numpy.ndarray
        Classification (``True`` if the component is a CSF one).
    """
    return x > thr_csf


def hplane_criteria(x, hplane=HYPERPLANE):
    """
    Compute the hyperplane criteria for classification.

    Parameters
    ----------
    x : numpy.ndarray
        Projection of edge & max_RP_corr feature scores to new 1D space.

    Returns
    -------
    :obj:`pandas.DataFrame`
        Features table with additional column "classification".

    """
    return (hplane[0] + np.dot(x, hplane[1:])) > 0


def predict(X, thr_csf=THR_CSF, thr_hfc=THR_HFC, hplane=HYPERPLANE, metric_metadata=None):
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
    proj = hplane_criteria(X[["max_RP_corr", "edge_fract"]].values, hplane=hplane)

    # Compute the CSF criteria
    csf = csf_criteria(X["csf_fract"].values, thr_csf=thr_csf)

    # Compute the HFC criteria
    hfc = hfc_criteria(X["HFC"].values, thr_hfc=thr_hfc)

    # Combine the criteria
    classification = csf | hfc | proj

    #  Turn classification into a list of string labels with rejected if true, accepted if false
    classification = ["rejected" if c else "accepted" for c in classification]

    # Classify the ICs
    return classification
