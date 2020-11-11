"""Utility functions for ICA-AROMA."""
import logging
import os
import os.path as op
import shutil

import nibabel as nib
import numpy as np
from nilearn import image, masking
from nilearn._utils import load_niimg
from scipy import stats
from tedana.decomposition import tedica, ma_pca
from tedana.utils import get_spectrum

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
    in_file = load_niimg(in_file)
    mask = load_niimg(mask)
    in_data = masking.apply_mask(in_file, mask)
    n_vols, n_voxels = in_data.shape

    # Start with a PCA to determine number of components and
    # dimensionally reduce data
    voxel_comp_weights, varex, varex_norm, mixing_pca = ma_pca(
            in_file, mask, criteria="mdl")
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


def register2MNI(fsl_dir, in_file, out_file, affmat, warp):
    """Register an image (or time-series of images) to MNI152 T1 2mm.

    If no affmat is defined, it only warps (i.e. it assumes that the data has
    been registered to the structural scan associated with the warp-file
    already). If no warp is defined either, it only resamples the data to 2mm
    isotropic if needed (i.e. it assumes that the data has been registered to
    a MNI152 template). In case only an affmat file is defined, it assumes that
    the data has to be linearly registered to MNI152 (i.e. the user has a
    reason not to use non-linear registration on the data).

    Parameters
    ----------
    fsl_dir : str
        Full path of the bin-directory of FSL
    in_file : str
        Full path to the data file (nii.gz) which has to be registerd to
        MNI152 T1 2mm
    out_file : str
        Full path of the output file
    affmat : str
        Full path of the mat file describing the linear registration (if data
        is still in native space)
    warp : str
        Full path of the warp file describing the non-linear registration (if
        data has not been registered to MNI152 space yet)

    Output
    ------
    melodic_IC_mm_MNI2mm.nii.gz : merged file containing the mixture modeling
                                  thresholded Z-statistical maps registered to
                                  MNI152 2mm
    """
    # Define the MNI152 T1 2mm template
    fslnobin = fsl_dir.rsplit('/', 2)[0]
    ref = op.join(fslnobin, 'data', 'standard', 'MNI152_T1_2mm_brain.nii.gz')

    # If the no affmat- or warp-file has been specified, assume that the data
    # is already in MNI152 space. In that case only check if resampling to
    # 2mm is needed
    if not affmat and not warp:
        in_img = load_niimg(in_file)
        # Get 3D voxel size
        pixdim1, pixdim2, pixdim3 = in_img.header.get_zooms()[:3]

        # If voxel size is not 2mm isotropic, resample the data, otherwise
        # copy the file
        if (pixdim1 != 2) or (pixdim2 != 2) or (pixdim3 != 2):
            resampled_img = image.resample_to_img(in_img, target_img=ref, interpolation="linear")
            resampled_img.to_filename(out_file)
        else:
            os.copyfile(in_file, out_file)

    # If only a warp-file has been specified, assume that the data has already
    # been registered to the structural scan. In that case apply the warping
    # without a affmat
    elif not affmat and warp:
        # Apply warp
        os.system(' '.join([op.join(fsl_dir, 'applywarp'),
                            '--ref=' + ref,
                            '--in=' + in_file,
                            '--out=' + out_file,
                            '--warp=' + warp,
                            '--interp=trilinear']))

    # If only a affmat-file has been specified perform affine registration to
    # MNI
    elif affmat and not warp:
        os.system(' '.join([op.join(fsl_dir, 'flirt'),
                            '-ref ' + ref,
                            '-in ' + in_file,
                            '-out ' + out_file,
                            '-applyxfm -init ' + affmat,
                            '-interp trilinear']))

    # If both a affmat- and warp-file have been defined, apply the warping
    # accordingly
    else:
        os.system(' '.join([op.join(fsl_dir, 'applywarp'),
                            '--ref=' + ref,
                            '--in=' + in_file,
                            '--out=' + out_file,
                            '--warp=' + warp,
                            '--premat=' + affmat,
                            '--interp=trilinear']))


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


def classification(out_dir, max_RP_corr, edge_fract, HFC, csf_fract):
    """Classify components as motion or non-motion based on four features.

    The four features used for classification are: maximum RP correlation,
    high-frequency content, edge-fraction, and CSF-fraction.

    Parameters
    ----------
    out_dir : str
        Full path of the output directory
    max_RP_corr : (C,) array_like
        Array of the 'maximum RP correlation' feature scores of the components
    edge_fract : (C,) array_like
        Array of the 'edge fraction' feature scores of the components
    HFC : (C,) array_like
        Array of the 'high-frequency content' feature scores of the components
    csf_fract : (C,) array_like
        Array of the 'CSF fraction' feature scores of the components

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
    # Classify the ICs as motion or non-motion

    # Define criteria needed for classification (thresholds and
    # hyperplane-parameters)
    thr_csf = 0.10
    thr_HFC = 0.35
    hyp = np.array([-19.9751070082159, 9.95127547670627, 24.8333160239175])

    # Project edge & max_RP_corr feature scores to new 1D space
    x = np.array([max_RP_corr, edge_fract])
    proj = hyp[0] + np.dot(x.T, hyp[1:])

    # Classify the ICs
    motion_ICs = np.squeeze(
        np.array(
            np.where(
                (proj > 0)
                + (csf_fract > thr_csf)
                + (HFC > thr_HFC)
            )
        )
    )

    # Put the feature scores in a text file
    np.savetxt(op.join(out_dir, 'feature_scores.txt'),
               np.vstack((max_RP_corr, edge_fract, HFC, csf_fract)).T)

    # Put the indices of motion-classified ICs in a text file
    with open(op.join(out_dir, 'classified_motion_ICs.txt'), 'w') as fo:
        if motion_ICs.size > 1:
            fo.write(','.join(['{:.0f}'.format(num) for num in
                               (motion_ICs + 1)]))
        elif motion_ICs.size == 1:
            fo.write('{:.0f}'.format(motion_ICs + 1))

    # Create a summary overview of the classification
    with open(op.join(out_dir, 'classification_overview.txt'), 'w') as fo:
        fo.write('\t'.join(['IC',
                            'Motion/noise',
                            'maximum RP correlation',
                            'Edge-fraction',
                            'High-frequency content',
                            'CSF-fraction']))
        fo.write('\n')
        for i in range(0, len(csf_fract)):
            if (proj[i] > 0) or (csf_fract[i] > thr_csf) or (HFC[i] > thr_HFC):
                classif = "True"
            else:
                classif = "False"
            fo.write('\t'.join(['{:d}'.format(i + 1),
                                classif,
                                '{:.2f}'.format(max_RP_corr[i]),
                                '{:.2f}'.format(edge_fract[i]),
                                '{:.2f}'.format(HFC[i]),
                                '{:.2f}'.format(csf_fract[i])]))
            fo.write('\n')

    return motion_ICs


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

    nonaggr_denoised_file = op.join(out_dir, "denoised_func_data_nonaggr.nii.gz")
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
            "  - None of the components were classified as motion, so no "
            "denoising is applied (the input file is copied as-is)."
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
