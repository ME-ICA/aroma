"""Functions to calculate ICA-AROMA features for component classification."""
import logging
import os
import sys

import nibabel as nib
import numpy as np
from nilearn import image, masking

from .utils import cross_correlation, get_resource_path

LGR = logging.getLogger(__name__)


def feature_time_series(mel_mix, mc):
    """Extract maximum motion parameter correlation scores from components.

    This function determines the maximum robust correlation of each component
    time series with a model of 72 realignment parameters.

    Parameters
    ----------
    mel_mix : str
        Full path of the melodic_mix text file.
        Stored array is (time x component).
    mc : str
        Full path of the text file containing the realignment parameters.
        Motion parameters are (time x 6), with the first three columns being
        rotation parameters (in radians) and the final three being translation
        parameters (in mm).

    Returns
    -------
    max_RP_corr : array_like
        Array of the maximum RP correlation feature scores for the components
        of the melodic_mix file.
    """
    # Read melodic mix file (IC time-series), subsequently define a set of
    # squared time-series
    mix = np.loadtxt(mel_mix)

    # Read motion parameter file
    rp6 = np.loadtxt(mc)

    # Determine the derivatives of the RPs (add zeros at time-point zero)
    _, nparams = rp6.shape
    rp6_der = np.vstack((
        np.zeros(nparams),
        np.diff(rp6, axis=0)
    ))

    # Create an RP-model including the RPs and its derivatives
    rp12 = np.hstack((rp6, rp6_der))

    # add the fw and bw shifted versions
    rp12_1fw = np.vstack((
        np.zeros(2 * nparams),
        rp12[:-1]
    ))
    rp12_1bw = np.vstack((
        rp12[1:],
        np.zeros(2 * nparams)
    ))
    rp_model = np.hstack((rp12, rp12_1fw, rp12_1bw))

    # Determine the maximum correlation between RPs and IC time-series
    nsplits = 1000
    nmixrows, nmixcols = mix.shape
    nrows_to_choose = int(round(0.9 * nmixrows))

    # Max correlations for multiple splits of the dataset (for a robust
    # estimate)
    max_correls = np.empty((nsplits, nmixcols))
    for i in range(nsplits):
        # Select a random subset of 90% of the dataset rows
        # (*without* replacement)
        if "pytest" in sys.modules: # detects we are using pytest
            np.random.seed(i)
        chosen_rows = np.random.choice(a=range(nmixrows),
                                       size=nrows_to_choose,
                                       replace=False)

        # Combined correlations between RP and IC time-series, squared and
        # non squared
        correl_nonsquared = cross_correlation(mix[chosen_rows],
                                              rp_model[chosen_rows])
        correl_squared = cross_correlation(mix[chosen_rows]**2,
                                           rp_model[chosen_rows]**2)
        correl_both = np.hstack((correl_squared, correl_nonsquared))

        # Maximum absolute temporal correlation for every IC
        max_correls[i] = np.abs(correl_both).max(axis=1)

    # Feature score is the mean of the maximum correlation over all the random
    # splits
    # Avoid propagating occasional nans that arise in artificial test cases
    max_RP_corr = np.nanmean(max_correls, axis=0)
    return max_RP_corr


def feature_frequency(mel_FT_mix, TR):
    """Extract the high-frequency content feature scores.

    This function determines the frequency, as fraction of the Nyquist
    frequency, at which the higher and lower frequencies explain half
    of the total power between 0.01Hz and Nyquist.

    Parameters
    ----------
    mel_FT_mix : str
        Full path of the melodic_FTmix text file.
        Stored array is (frequency x component), with frequencies
        ranging from 0 Hz to Nyquist frequency.
    TR : float
        TR (in seconds) of the fMRI data

    Returns
    -------
    HFC : array_like
        Array of the HFC ('High-frequency content') feature scores
        for the components of the melodic_FTmix file
    """
    # Determine sample frequency
    Fs = 1 / TR

    # Determine Nyquist-frequency
    Ny = Fs / 2

    # Load melodic_FTmix file
    FT = np.loadtxt(mel_FT_mix)
    n_frequencies = FT.shape[0]

    # Determine which frequencies are associated with every row in the
    # melodic_FTmix file (assuming the rows range from 0Hz to Nyquist)
    f = Ny * np.arange(1, n_frequencies + 1) / n_frequencies

    # Only include frequencies higher than 0.01Hz
    fincl = np.squeeze(np.array(np.where(f > 0.01)))
    FT = FT[fincl, :]
    f = f[fincl]

    # Set frequency range to [0-1]
    f_norm = (f - 0.01) / (Ny - 0.01)

    # For every IC; get the cumulative sum as a fraction of the total sum
    fcumsum_fract = np.cumsum(FT, axis=0) / np.sum(FT, axis=0)

    # Determine the index of the frequency with the fractional cumulative sum
    # closest to 0.5
    idx_cutoff = np.argmin(np.abs(fcumsum_fract - 0.5), axis=0)

    # Now get the fractions associated with those indices index, these are the
    # final feature scores
    HFC = f_norm[idx_cutoff]

    # Return feature score
    return HFC


def feature_spatial(mel_IC):
    """Extract the spatial feature scores.

    For each IC it determines the fraction of the mixture modeled thresholded
    Z-maps respectively located within the CSF or at the brain edges,
    using predefined standardized masks.

    Parameters
    ----------
    mel_IC : str
        Full path of the nii.gz file containing mixture-modeled thresholded
        (p<0.5) Z-maps, registered to the MNI152 2mm template

    Returns
    -------
    edge_fract : array_like
        Array of the edge fraction feature scores for the components of the
        mel_IC file
    csf_fract : array_like
        Array of the CSF fraction feature scores for the components of the
        mel_IC file
    """
    # Get the number of ICs
    mel_IC_img = nib.load(mel_IC)
    num_ICs = mel_IC_img.shape[3]

    masks_dir = get_resource_path()
    csf_mask = os.path.join(masks_dir, "mask_csf.nii.gz")
    edge_mask = os.path.join(masks_dir, "mask_edge.nii.gz")
    out_mask = os.path.join(masks_dir, "mask_out.nii.gz")

    # Loop over ICs
    edge_fract = np.zeros(num_ICs)
    csf_fract = np.zeros(num_ICs)
    for i in range(num_ICs):
        # Extract IC from the merged melodic_IC_thr2MNI2mm file
        temp_IC = image.index_img(mel_IC, i)

        # Change to absolute Z-values
        temp_IC = image.math_img("np.abs(img)", img=temp_IC)

        # Get sum of Z-values within the total Z-map (calculate via the mean
        # and number of non-zero voxels)
        temp_IC_data = temp_IC.get_fdata()
        tot_sum = np.sum(temp_IC_data)

        if tot_sum == 0:
            LGR.info("\t- The spatial map of component {} is empty. "
                     "Please check!".format(i + 1))

        # Get sum of Z-values of the voxels located within the CSF
        # (calculate via the mean and number of non-zero voxels)
        csf_data = masking.apply_mask(temp_IC, csf_mask)
        csf_sum = np.sum(csf_data)

        # Get sum of Z-values of the voxels located within the Edge
        # (calculate via the mean and number of non-zero voxels)
        edge_data = masking.apply_mask(temp_IC, edge_mask)
        edge_sum = np.sum(edge_data)

        # Get sum of Z-values of the voxels located outside the brain
        # (calculate via the mean and number of non-zero voxels)
        out_data = masking.apply_mask(temp_IC, out_mask)
        out_sum = np.sum(out_data)

        # Determine edge and CSF fraction
        if tot_sum != 0:
            edge_fract[i] = (out_sum + edge_sum) / (tot_sum - csf_sum)
            csf_fract[i] = csf_sum / tot_sum
        else:
            edge_fract[i] = 0
            csf_fract[i] = 0

    # Return feature scores
    return edge_fract, csf_fract
