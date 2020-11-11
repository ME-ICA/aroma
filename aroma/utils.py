# CHANGES
# -------
# Log of changes as mandated by the original Apache 2.0 License of ICA-AROMA
#
#   * Drop ``runICA`` and ``register2MNI`` functions
#   * Base ``classifier`` on Pandas, and revise signature (``predict(X)``)
#     to make it more similar to scikit learn
#   * Return classification labels directly on ``predict``
#
"""Utility functions for ICA-AROMA."""
import logging
import os.path as op
import shutil

import nibabel as nib
import numpy as np
from nilearn import masking

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


def cross_correlation(a, b):
    """Perform cross-correlations between columns of two matrices.

    Parameters
    ----------
    a : (M x X) array_like
        First array to cross-correlate
    b : (N x X) array_like
        Second array to cross-correlate

    Returns
    -------
    correlations : (M x N) array_like
        Cross-correlations of columns of a against columns of b.
    """
    assert a.ndim == b.ndim == 2
    _, ncols_a = a.shape
    # nb variables in columns rather than rows hence transpose
    # extract just the cross terms between cols in a and cols in b
    return np.corrcoef(a.T, b.T)[:ncols_a, ncols_a:]


def denoising(fsl_dir, in_file, out_dir, mixing, den_type, den_idx):
    """Remove noise components from fMRI data.

    Parameters
    ----------
    fsl_dir : str
        Full path of the bin-directory of FSL
    in_file : str
        Full path to the data file (nii.gz) which has to be denoised
    out_dir : str
        Full path of the output directory
    mixing : str
        Full path of the melodic_mix text file
    den_type : {"aggr", "nonaggr", "both"}
        Type of requested denoising ('aggr': aggressive, 'nonaggr':
        non-aggressive, 'both': both aggressive and non-aggressive
    den_idx : array_like
        Index of the components that should be regressed out

    Output
    ------
    denoised_func_data_<den_type>.nii.gz : The denoised fMRI data
    """
    # Check if denoising is needed (i.e. are there motion components?)
    motion_components_found = den_idx.size > 0

    nonaggr_denoised_file = op.join(out_dir,
                                    "denoised_func_data_nonaggr.nii.gz")
    aggr_denoised_file = op.join(out_dir, "denoised_func_data_aggr.nii.gz")

    if motion_components_found:
        mixing = np.loadtxt(mixing)
        motion_components = mixing[:, den_idx]

        # Create a fake mask to make it easier to reshape the full data to 2D
        img = nib.load(in_file)
        full_mask = nib.Nifti1Image(np.ones(img.shape[:3], int), img.affine)
        data = masking.apply_mask(img, full_mask)  # T x S

        # Non-aggressive denoising of the data using fsl_regfilt
        # (partial regression), if requested
        if den_type in ("nonaggr", "both"):
            # Fit GLM to all components
            betas = np.linalg.lstsq(mixing, data, rcond=None)[0]

            # Denoise the data using the betas from just the bad components.
            pred_data = np.dot(motion_components, betas[den_idx, :])
            data_denoised = data - pred_data

            # Save to file.
            img_denoised = masking.unmask(data_denoised, full_mask)
            img_denoised.to_filename(nonaggr_denoised_file)

        # Aggressive denoising of the data using fsl_regfilt (full regression)
        if den_type in ("aggr", "both"):
            # Denoise the data with the bad components.
            betas = np.linalg.lstsq(motion_components, data, rcond=None)[0]
            pred_data = np.dot(motion_components, betas)
            data_denoised = data - pred_data

            # Save to file.
            img_denoised = masking.unmask(data_denoised, full_mask)
            img_denoised.to_filename(aggr_denoised_file)
    else:
        LGR.warning(
                    "  - None of the components were classified as motion, "
                    "so no denoising is applied (the input file is copied "
                    "as-is)."
        )
        if den_type in ("nonaggr", "both"):
            shutil.copyfile(in_file, nonaggr_denoised_file)

        if den_type in ("aggr", "both"):
            shutil.copyfile(in_file, aggr_denoised_file)


def get_resource_path():
    """Return the path to general resources.

    Returns the path to general resources, terminated with separator.
    Resources are kept outside package folder in "resources".
    Based on function by Yaroslav Halchenko used in Neurosynth Python package.

    Returns
    -------
    resource_path : str
        Absolute path to resources folder.
    """
    return op.abspath(op.join(op.dirname(__file__), "resources") + op.sep)
