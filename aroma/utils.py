"""Utility functions for ICA-AROMA."""
import json
import logging
import os.path as op
import shutil
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, binary_dilation
from skimage.morphology import ball
from nilearn import masking
from nilearn._utils import load_niimg, copy_img

LGR = logging.getLogger(__name__)


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


def classification(features_df, metric_metadata=None):
    """Classify components as motion or non-motion based on four features.

    The four features used for classification are: maximum RP correlation,
    high-frequency content, edge-fraction, and CSF-fraction.

    Parameters
    ----------
    features_df : (C x 4) :obj:`pandas.DataFrame`
        DataFrame with the following columns:
        "edge_fract", "csf_fract", "max_RP_corr", and "HFC".
    metric_metadata : :obj:`dict` or None, optional
        Metric metadata in a dictionary.

    Returns
    -------
    features_df
    metric_metadata
    """
    # Define criteria needed for classification (thresholds and
    # hyperplane-parameters)
    THR_CSF = 0.10
    THR_HFC = 0.35
    HYPERPLANE = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

    if isinstance(metric_metadata, dict):
        metric_metadata["classification"] = {
            "LongName": "Component classification",
            "Description": ("Classification from the classification procedure."),
            "Levels": {
                "accepted": "A component that is determined not to be associated with motion.",
                "rejected": "A motion-related component.",
            },
        }
        metric_metadata["rationale"] = {
            "LongName": "Rationale for component classification",
            "Description": (
                "The reason for the classification. "
                "In cases where components are classified based on more than one criterion, "
                "they are listed sequentially, separated by semicolons."
            ),
            "Levels": {
                "CSF": f"The csf_fract value is higher than {THR_CSF}",
                "HFC": f"The HFC value is higher than {THR_HFC}",
                "hyperplane": (
                    "After the max_RP_corr and edge_fract values are projected "
                    "to a hyperplane, the projected point is less than zero."
                ),
            },
        }

    # Classify the ICs as motion (rejected) or non-motion (accepted)
    features_df["classification"] = "accepted"
    features_df["rationale"] = ""

    # CSF
    rej_csf = features_df["csf_fract"] > THR_CSF
    features_df.loc[rej_csf, "classification"] = "rejected"
    features_df.loc[rej_csf, "rationale"] += "CSF;"

    # HFC
    rej_hfc = features_df["HFC"] > THR_HFC
    features_df.loc[rej_hfc, "classification"] = "rejected"
    features_df.loc[rej_hfc, "rationale"] += "HFC;"

    # Hyperplane
    # Project edge & max_RP_corr feature scores to new 1D space
    x = features_df[["max_RP_corr", "edge_fract"]].values
    proj = HYPERPLANE[0] + np.dot(x, HYPERPLANE[1:])
    rej_hyperplane = proj > 0
    features_df.loc[rej_hyperplane, "classification"] = "rejected"
    features_df.loc[rej_hyperplane, "rationale"] += "hyperplane;"

    # Classify the ICs
    is_motion = (features_df["csf_fract"] > THR_CSF) | (features_df["HFC"] > THR_HFC) | (proj > 0)
    assert np.array_equal(is_motion, (features_df["classification"] == "rejected").values)

    # Reorder columns and remove trailing semicolons
    features_df = clean_dataframe(features_df)

    return features_df, metric_metadata


def write_metrics(features_df, out_dir, metric_metadata=None):
    """Write out feature/classification information and metadata.

    Parameters
    ----------
    features_df : (C x 5) :obj:`pandas.DataFrame`
        DataFrame with metric values and classifications.
        Must have the following columns: "edge_fract", "csf_fract", "max_RP_corr", "HFC", and
        "classification".
    out_dir : :obj:`str`
        Output directory.
    metric_metadata : :obj:`dict` or None, optional
        Metric metadata in a dictionary.

    Returns
    -------
    motion_ICs : array_like
        Array containing the indices of the components identified as motion components.

    Output
    ------
    AROMAnoiseICs.csv : A text file containing the indices of the
                        components identified as motion components
    desc-AROMA_metrics.tsv
    desc-AROMA_metrics.json
    """
    # Put the indices of motion-classified ICs in a text file (starting with 1)
    motion_ICs = features_df["classification"][features_df["classification"] == "rejected"].index
    motion_ICs = motion_ICs.values

    with open(op.join(out_dir, "AROMAnoiseICs.csv"), "w") as fo:
        out_str = ",".join(motion_ICs.astype(str))
        fo.write(out_str)

    # Create a summary overview of the classification
    out_file = op.join(out_dir, "desc-AROMA_metrics.tsv")
    features_df.to_csv(out_file, sep="\t", index_label="IC")

    if isinstance(metric_metadata, dict):
        with open(op.join(out_dir, "desc-AROMA_metrics.json"), "w") as fo:
            json.dump(metric_metadata, fo, sort_keys=True, indent=4)

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
    desc-smoothAROMA<den_type>_bold.nii.gz : The denoised fMRI data
    """
    # Check if denoising is needed (i.e. are there motion components?)
    motion_components_found = den_idx.size > 0

    nonaggr_denoised_file = op.join(out_dir, "desc-smoothAROMAnonaggr_bold.nii.gz")
    aggr_denoised_file = op.join(out_dir, "desc-smoothAROMAaggr_bold.nii.gz")

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
            "  - None of the components were classified as motion, "
            "so no denoising is applied (the input file is copied "
            "as-is)."
        )
        if den_type in ("nonaggr", "both"):
            shutil.copyfile(in_file, nonaggr_denoised_file)

        if den_type in ("aggr", "both"):
            shutil.copyfile(in_file, aggr_denoised_file)


def clean_dataframe(comptable):
    """Reorder columns in component table.

    This places "rationale" and "classification" at the end and
    removes trailing semicolons from rationale column.
    """
    cols_at_end = ["classification", "rationale"]
    comptable = comptable[
        [c for c in comptable if c not in cols_at_end] + [c for c in cols_at_end if c in comptable]
    ]
    comptable["rationale"] = comptable["rationale"].str.rstrip(";")
    return comptable


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
    motpars_fsl = confounds[["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]].values
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
            raise Exception("Motion parameter source could not be determined automatically.")

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

def csf_mask_from_T2star(T2star_map:str, brain_mask:str, T2star_thresh:float= 0.080, erode_voxels:int=2):
    """"Create mask of ventricle CSF based on T2-star map. 

    Parameters
    ----------
    T2star_map: str
        path to nifti file with T2star map.
    brain_mask: str
        path to nifti file with brain mask
    T2star_thresh: :obj:`float`
        mask will only consider voxels with T2-star values
        above T2star_thresh. Default value of 0.080 sec assumes 3T 
        acquisition.
    erode_voxels: :obj:`int``
        number of voxels to erode (radius of ball)

    Returns
    -------
    csf_mask: path to nifti volume with binary mask of ventricle CSF.
    """

    T2star_img = load_niimg(T2star_map)
    brain_mask_img = load_niimg(brain_mask).dataobj > 0.5

    # only keep CSF voxels in the ventricles by applying eroded brain mask
    eroded_brain_mask = binary_erosion(brain_mask_img, structure=ball(erode_voxels)) 
    # TODO: pass number of iterations as input argument

    # save to nifti file
    temp_dir = tempfile.mkdtemp()
    path_csf_mask = op.join(temp_dir,"mask_csf.nii.gz")

    # Apply threshold to identify CSF voxels
    CSF_mask = nib.Nifti1Image((T2star_img.get_fdata() > T2star_thresh) * eroded_brain_mask, 
        T2star_img.affine, header=T2star_img.header)
    CSF_mask.to_filename(path_csf_mask)

    return path_csf_mask


def edge_out_mask(brain_mask:str, erode_edge_voxels:int=2, dilate_out_voxels:int=0):
    """"Create mask of edge voxels and out-of-the-brain voxels. 
        The edge mask will be the subtraction of the brain mask minus
        this mask eroded by erode_voxels, whereas the out_mask will
        be all the voxels outside the brain mask, even though the user
        can specify to dilate the brain mask (e.g., in cases of inaccurate 
        skull stripping)

    Parameters
    ----------
    brain_img: path to nifti file with a brain to compute the within-brain mask
    erode_edge_voxels: :obj:`int`
        Number of voxels to erode in order to compute the edge mask.
    dilate_out_voxels: :obj:`int``
        Number of voxels to erode in order to compute the out-of-brain mask

    Returns
    -------
    edge_mask: path to nifti volume with binary mask of edge-brain voxels.
    out_mask: path to nifti volume wiht binary mask of out-of-brain voxels.
    """

    brain_mask_img = load_niimg(brain_mask)
    brain_mask_bool = brain_mask_img.dataobj > 0.5

    # erode brain mask to compute edge-brain mask
    brain_mask_eroded = binary_erosion(brain_mask_bool, structure=ball(erode_edge_voxels)).astype(bool)
    brain_edge_bool = brain_mask_bool.copy()
    brain_edge_bool[brain_mask_eroded] = False

    if dilate_out_voxels > 0:
        out_mask_bool = binary_dilation(brain_mask_bool, structure=ball(dilate_out_voxels)) 

    # save to nifti file
    temp_dir = tempfile.mkdtemp()
    path_edge_mask = op.join(temp_dir,"mask_edge.nii.gz")
    path_out_mask = op.join(temp_dir,"mask_out.nii.gz")

    # Compute masks
    edge_mask = nib.Nifti1Image(brain_edge_bool, 
        brain_mask_img.affine, header=brain_mask_img.header)
    out_mask = nib.Nifti1Image(out_mask_bool, 
        brain_mask_img.affine, header=brain_mask_img.header)
    edge_mask.to_filename(path_edge_mask)
    out_mask.to_filename(path_out_mask)

    return path_edge_mask, path_out_mask