"""Utility functions for ICA-AROMA."""
import logging
import os.path as op
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import masking
from nilearn._utils import load_niimg
from scipy import stats

from .mixture import GGM

LGR = logging.getLogger(__name__)


def run_ica(in_file, mask, n_components=-1, t_r=None):
    """Run ICA and collect relevant outputs.

    Parameters
    ----------
    in_file : str
        Full path to the fMRI data file (nii.gz) on which ICA
        should be run
    mask : str
        Full path of the mask to be applied during ICA
    n_components : int, optional
        Dimensionality of ICA.
        If -1, then dimensionality will be automatically detected.
        Default is -1.
    t_r : float or None, optional
        Repetition time (TR), in seconds, of the fMRI data.
        If None, then TR will be inferred from the data file's header.
        Default is None.

    Returns
    -------
    components_img_z_thresh : 4D img_like
    mixing_ica : (T x C) array_like
    mixing_power_spectra : (F x C) array_like
    """
    from mapca import ma_pca
    from tedana.decomposition import tedica

    in_file = load_niimg(in_file)
    mask = load_niimg(mask)
    in_data = masking.apply_mask(in_file, mask)
    n_vols, n_voxels = in_data.shape

    # Start with a PCA to determine number of components and
    # dimensionally reduce data
    voxel_comp_weights, varex, varex_norm, mixing_pca = ma_pca(in_file, mask, criteria="mdl")
    n_components = len(varex)
    # kept_data is SxT, instead of nilearn-standard TxS
    voxel_kept_comp_weighted = voxel_comp_weights * varex[None, :]
    kept_data = np.dot(voxel_kept_comp_weighted, mixing_pca.T)
    kept_data = stats.zscore(kept_data, axis=1)  # variance normalize time series
    kept_data = stats.zscore(kept_data, axis=None)  # variance normalize everything
    assert kept_data.shape == (n_voxels, n_vols)
    LGR.info("{} components retained by PCA".format(n_components))

    mixing_ica = tedica(kept_data, n_components, fixed_seed=1, maxit=500, maxrestart=5)
    assert mixing_ica.shape == (n_vols, n_components)

    # Compute component maps
    data_z = stats.zscore(in_data, axis=0)
    mixing_z = stats.zscore(mixing_ica, axis=0)
    components_arr_z = np.linalg.lstsq(mixing_z, data_z, rcond=None)[0].T
    assert components_arr_z.shape == (n_voxels, n_components)
    # compute skews to determine signs based on unnormalized weights,
    # correct mixing matrix & component map signs based on spatial distribution tails
    signs = stats.skew(components_arr_z, axis=0)
    signs /= np.abs(signs)
    mixing_z = mixing_z * signs
    components_arr_z *= signs

    THRESH = 0.5
    # Preallocate arrays
    components_arr_z_thresh = np.zeros(components_arr_z.shape)
    mixing_power_spectra = []
    for i_comp in range(components_arr_z.shape[1]):
        # Mixture modeling
        component_arr = components_arr_z[:, i_comp]
        ggm = GGM()
        ggm.estimate(component_arr, niter=1000)
        gauss_probs, gamma_probs = ggm.posterior(component_arr)
        # apply threshold
        component_arr[gamma_probs < THRESH] = 0
        components_arr_z_thresh[:, i_comp] = component_arr

        # Now get the FT array
        # TODO: Check that (1) freqs are same and (2) range from 0 to Nyquist
        spectrum, freqs = get_spectrum(mixing_ica[:, i_comp], t_r)
        mixing_power_spectra.append(spectrum)
    components_img_z_thresh = masking.unmask(components_arr_z_thresh.T, mask)
    mixing_power_spectra = np.stack(mixing_power_spectra, axis=-1)
    assert mixing_power_spectra.shape == (len(freqs), n_components), mixing_power_spectra.shape

    return components_img_z_thresh, mixing_ica, mixing_power_spectra


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
    if a.ndim != 2:
        raise ValueError(f"Input `a` must be 2D, not {a.ndim}D")

    if b.ndim != 2:
        raise ValueError(f"Input `b` must be 2D, not {b.ndim}D")

    _, ncols_a = a.shape
    # nb variables in columns rather than rows hence transpose
    # extract just the cross terms between cols in a and cols in b
    return np.corrcoef(a.T, b.T)[:ncols_a, ncols_a:]


def classification(features_df, out_dir):
    """Classify components as motion or non-motion based on four features.

    The four features used for classification are: maximum RP correlation,
    high-frequency content, edge-fraction, and CSF-fraction.

    Parameters
    ----------
    features_df : (C x 4) pandas.DataFrame
        DataFrame with the following columns:
        "edge_fract", "csf_fract", "max_RP_corr", and "HFC".
    out_dir : str
        Full path of the output directory

    Returns
    -------
    motion_ICs : array_like
        Array containing the indices of the components identified as motion
        components

    Output
    ------
    classified_motion_ICs.txt : A text file containing the indices of the
                                components identified as motion components
    """
    # Put the feature scores in a text file
    features_df.to_csv(op.join(out_dir, "feature_scores.tsv"), sep="\t", index=False)

    # Classify the ICs as motion or non-motion

    # Define criteria needed for classification (thresholds and
    # hyperplane-parameters)
    THR_CSF = 0.10
    THR_HFC = 0.35
    HYPERPLANE = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

    # Project edge & max_RP_corr feature scores to new 1D space
    x = features_df[["max_RP_corr", "edge_fract"]].values
    proj = HYPERPLANE[0] + np.dot(x, HYPERPLANE[1:])

    # Classify the ICs
    is_motion = (
        (features_df["csf_fract"] > THR_CSF)
        | (features_df["HFC"] > THR_HFC)
        | (proj > 0)
    )
    features_df["classification"] = is_motion
    features_df["classification"] = features_df["classification"].map(
        {True: "rejected", False: "accepted"}
    )

    # Put the indices of motion-classified ICs in a text file (starting with 1)
    motion_ICs = features_df["classification"][features_df["classification"] == "rejected"].index
    motion_ICs = motion_ICs.values
    with open(op.join(out_dir, "classified_motion_ICs.txt"), "w") as fo:
        out_str = ",".join(motion_ICs.astype(str))
        fo.write(out_str)

    # Create a summary overview of the classification
    features_df.to_csv(
        op.join(out_dir, "classification_overview.txt"), sep="\t", index_label="IC"
    )

    return motion_ICs


def denoising(in_file, out_dir, mixing, den_type, den_idx):
    """Remove noise components from fMRI data.

    Parameters
    ----------
    in_file : str
        Full path to the data file (nii.gz) which has to be denoised
    out_dir : str
        Full path of the output directory
    mixing : numpy.ndarray of shape (T, C)
        Mixing matrix.
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

    nonaggr_denoised_file = op.join(out_dir, "denoised_func_data_nonaggr.nii.gz")
    aggr_denoised_file = op.join(out_dir, "denoised_func_data_aggr.nii.gz")

    if motion_components_found:
        motion_components = mixing[:, den_idx]

        # Create a fake mask to make it easier to reshape the full data to 2D
        img = load_niimg(in_file)
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
            "  - None of the components were classified as motion, so no "
            "denoising is applied (the input file is copied as-is)."
        )
        if den_type in ("nonaggr", "both"):
            shutil.copyfile(in_file, nonaggr_denoised_file)

        if den_type in ("aggr", "both"):
            shutil.copyfile(in_file, aggr_denoised_file)


def motpars_fmriprep2fsl(confounds):
    """Convert fMRIPrep motion parameters to FSL format.

    Parameters
    ----------
    confounds : str or pandas.DataFrame
        Confounds data from fMRIPrep.
        Relevant columns have the format "[rot|trans]_[x|y|z]".
        Rotations are in radians.

    Returns
    -------
    motpars_fsl : (T x 6) numpy.ndarray
        Motion parameters in FSL format, with rotations first (in radians) and
        translations second.
    """
    if isinstance(confounds, str) and op.isfile(confounds):
        confounds = pd.read_table(confounds)
    elif not isinstance(confounds, pd.DataFrame):
        raise ValueError("Input must be an existing file or a DataFrame.")

    # Rotations are in radians
    motpars_fsl = confounds[
        ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    ].values
    return motpars_fsl


def motpars_spm2fsl(motpars):
    """Convert SPM format motion parameters to FSL format.

    Parameters
    ----------
    motpars : str or array_like
        SPM-format motion parameters.
        Rotations are in degrees and translations come first.

    Returns
    -------
    motpars_fsl : (T x 6) numpy.ndarray
        Motion parameters in FSL format, with rotations first (in radians) and
        translations second.
    """
    if isinstance(motpars, str) and op.isfile(motpars):
        motpars = np.loadtxt(motpars)
    elif not isinstance(motpars, np.ndarray):
        raise ValueError("Input must be an existing file or a numpy array.")

    if motpars.shape[1] != 6:
        raise ValueError(
            "Motion parameters must have exactly 6 columns, not {}.".format(motpars.shape[1])
        )

    # Split translations from rotations
    trans, rot = motpars[:, :3], motpars[:, 3:]

    # Convert rotations from degrees to radians
    rot *= np.pi / 180.0

    # Place rotations first
    motpars_fsl = np.hstack((rot, trans))
    return motpars_fsl


def motpars_afni2fsl(motpars):
    """Convert AFNI format motion parameters to FSL format.

    Parameters
    ----------
    motpars : str or array_like
        AfNI-format motion parameters in 1D file.
        Rotations are in degrees and translations come first.

    Returns
    -------
    motpars_fsl : (T x 6) numpy.ndarray
        Motion parameters in FSL format, with rotations first (in radians) and
        translations second.
    """
    if isinstance(motpars, str) and op.isfile(motpars):
        motpars = np.loadtxt(motpars)
    elif not isinstance(motpars, np.ndarray):
        raise ValueError("Input must be an existing file or a numpy array.")

    if motpars.shape[1] != 6:
        raise ValueError(
            "Motion parameters must have exactly 6 columns, not {}.".format(motpars.shape[1])
        )

    # Split translations from rotations
    trans, rot = motpars[:, :3], motpars[:, 3:]

    # Convert rotations from degrees to radians
    rot *= np.pi / 180.0

    # Place rotations first
    motpars_fsl = np.hstack((rot, trans))
    return motpars_fsl


def load_motpars(motion_file, source="auto"):
    """Load motion parameters from file.

    Parameters
    ----------
    motion_file : str
        Motion file.
    source : {"auto", "spm", "afni", "fsl", "fmriprep"}, optional
        Source of the motion data.
        If "auto", try to deduce the source based on the name of the file.

    Returns
    -------
    motpars : (T x 6) numpy.ndarray
        Motion parameters in FSL format, with rotations first (in radians) and
        translations second.
    """
    if source == "auto":
        if op.basename(motion_file).startswith("rp_") and motion_file.endswith(".txt"):
            source = "spm"
        elif motion_file.endswith(".1D"):
            source = "afni"
        elif motion_file.endswith(".tsv"):
            source = "fmriprep"
        elif motion_file.endswith(".txt"):
            source = "fsl"
        else:
            raise Exception(
                "Motion parameter source could not be determined automatically."
            )

    if source == "spm":
        motpars = motpars_spm2fsl(motion_file)
    elif source == "afni":
        motpars = motpars_afni2fsl(motion_file)
    elif source == "fsl":
        motpars = np.loadtxt(motion_file)
    elif source == "fmriprep":
        motpars = motpars_fmriprep2fsl(motion_file)
    else:
        raise ValueError('Source "{0}" not supported.'.format(source))

    return motpars


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


def get_spectrum(data: np.array, tr: float):
    """Return the power spectrum and corresponding frequencies of a time series.

    Parameters
    ----------
    data : numpy.ndarray of shape (T, C) or (T,)
        A time series of shape T (time) by C (component),
        on which you would like to perform an fft.
    tr : :obj:`float`
        Repetition time (TR) of the data, in seconds.

    Returns
    -------
    power_spectrum : numpy.ndarray of shape (F, C)
        Power spectrum of the input time series. C is component, F is frequency.
    freqs : numpy.ndarray of shape (F,)
        Frequencies corresponding to the columns of power_spectrum.
    """
    if data.ndim > 2:
        raise ValueError(f"Input `data` must be 1D or 2D, not {data.ndim}D")

    if data.ndim == 1:
        data = data[:, None]

    power_spectrum = np.abs(np.fft.rfft(data, axis=0)) ** 2
    freqs = np.fft.rfftfreq((power_spectrum.shape[0] * 2) - 1, tr)
    idx = np.argsort(freqs)
    return power_spectrum[idx, :], freqs[idx]
