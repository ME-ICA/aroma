"""Utility functions for ICA-AROMA."""
import os
import os.path as op
import shutil

import nibabel as nib
import numpy as np
from nilearn import image, masking


def runICA(fsl_dir, in_file, out_dir, mel_dir_in, mask, dim, TR):
    """Run MELODIC and merge the thresholded ICs into a single 4D nifti file.

    Parameters
    ----------
    fsl_dir : str
        Full path of the bin-directory of FSL
    in_file : str
        Full path to the fMRI data file (nii.gz) on which MELODIC
        should be run
    out_dir : str
        Full path of the output directory
    mel_dir_in : str or None
        Full path of the MELODIC directory in case it has been run
        before, otherwise None.
    mask : str
        Full path of the mask to be applied during MELODIC
    dim : int
        Dimensionality of ICA
    TR : float
        TR (in seconds) of the fMRI data

    Output
    ------
    melodic.ica/: MELODIC directory
    melodic_IC_thr.nii.gz: Merged file containing the mixture modeling
                           thresholded Z-statistical maps located in
                           melodic.ica/stats/
    """
    # Define the 'new' MELODIC directory and predefine some associated files
    mel_dir = op.join(out_dir, "melodic.ica")
    mel_IC = op.join(mel_dir, "melodic_IC.nii.gz")
    mel_IC_mix = op.join(mel_dir, "melodic_mix")
    mel_IC_thr = op.join(out_dir, "melodic_IC_thr.nii.gz")

    # When a MELODIC directory is specified,
    # check whether all needed files are present.
    # Otherwise... run MELODIC again
    if (
        mel_dir_in
        and op.isfile(op.join(mel_dir_in, "melodic_IC.nii.gz"))
        and op.isfile(op.join(mel_dir_in, "melodic_FTmix"))
        and op.isfile(op.join(mel_dir_in, "melodic_mix"))
    ):
        print("  - The existing/specified MELODIC directory will be used.")

        # If a 'stats' directory is present (contains thresholded spatial maps)
        # create a symbolic link to the MELODIC directory.
        # Otherwise create specific links and
        # run mixture modeling to obtain thresholded maps.
        if op.isdir(op.join(mel_dir_in, "stats")):
            os.symlink(mel_dir_in, mel_dir)
        else:
            print(
                "  - The MELODIC directory does not contain the required "
                "'stats' folder. Mixture modeling on the Z-statistical "
                "maps will be run."
            )

            # Create symbolic links to the items in the specified melodic
            # directory
            os.makedirs(mel_dir)
            for item in os.listdir(mel_dir_in):
                os.symlink(op.join(mel_dir_in, item), op.join(mel_dir, item))

            # Run mixture modeling
            melodic_command = (
                "{0} --in={1} --ICs={1} --mix={2} --out_dir={3} "
                "--0stats --mmthresh=0.5"
            ).format(
                op.join(fsl_dir, "melodic"),
                mel_IC,
                mel_IC_mix,
                mel_dir,
            )
            os.system(melodic_command)
    else:
        # If a melodic directory was specified, display that it did not
        # contain all files needed for ICA-AROMA (or that the directory
        # does not exist at all)
        if mel_dir_in:
            if not op.isdir(mel_dir_in):
                print(
                    "  - The specified MELODIC directory does not exist. "
                    "MELODIC will be run separately."
                )
            else:
                print(
                    "  - The specified MELODIC directory does not contain "
                    "the required files to run ICA-AROMA. MELODIC will be "
                    "run separately."
                )

        # Run MELODIC
        melodic_command = (
            "{0} --in={1} --outdir={2} --mask={3} --dim={4} "
            "--Ostats --nobet --mmthresh=0.5 --report "
            "--tr={5}"
        ).format(op.join(fsl_dir, "melodic"), in_file, mel_dir, mask, dim, TR)
        os.system(melodic_command)

    # Get number of components
    mel_IC_img = nib.load(mel_IC)
    nr_ICs = mel_IC_img.shape[3]

    # Merge mixture modeled thresholded spatial maps. Note! In case that
    # mixture modeling did not converge, the file will contain two spatial
    # maps. The latter being the results from a simple null hypothesis test.
    # In that case, this map will have to be used (first one will be empty).
    zstat_imgs = []
    for i in range(1, nr_ICs + 1):
        # Define thresholded zstat-map file
        z_temp = op.join(mel_dir, "stats", "thresh_zstat{0}.nii.gz".format(i))

        # Get number of volumes in component's thresholded image
        z_temp_img = nib.load(z_temp)
        if z_temp_img.ndim == 4:
            len_IC = z_temp_img.shape[3]
            # Extract last spatial map within the thresh_zstat file
            zstat_img = image.index_img(z_temp_img, len_IC - 1)
        else:
            zstat_img = z_temp_img

        zstat_imgs.append(zstat_img)

    # Merge to 4D
    zstat_4d_img = image.concat_imgs(zstat_imgs)

    # Apply the mask to the merged image (in case a melodic-directory was
    # predefined and run with a different mask)
    zstat_4d_img = image.math_img(
        "stat * mask[:, :, :, None]", stat=zstat_4d_img, mask=mask
    )
    zstat_4d_img.to_filename(mel_IC_thr)


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
    fslnobin = fsl_dir.rsplit("/", 2)[0]
    ref = op.join(fslnobin, "data", "standard", "MNI152_T1_2mm_brain.nii.gz")

    # If the no affmat- or warp-file has been specified, assume that the data
    # is already in MNI152 space. In that case only check if resampling to
    # 2mm is needed
    if not affmat and not warp:
        in_img = nib.load(in_file)
        # Get 3D voxel size
        pixdim1, pixdim2, pixdim3 = in_img.header.get_zooms()[:3]

        # If voxel size is not 2mm isotropic, resample the data, otherwise
        # copy the file
        if (pixdim1 != 2) or (pixdim2 != 2) or (pixdim3 != 2):
            os.system(
                " ".join(
                    [
                        op.join(fsl_dir, "flirt"),
                        " -ref " + ref,
                        " -in " + in_file,
                        " -out " + out_file,
                        " -applyisoxfm 2 -interp trilinear",
                    ]
                )
            )
        else:
            os.copyfile(in_file, out_file)

    # If only a warp-file has been specified, assume that the data has already
    # been registered to the structural scan. In that case apply the warping
    # without a affmat
    elif not affmat and warp:
        # Apply warp
        os.system(
            " ".join(
                [
                    op.join(fsl_dir, "applywarp"),
                    "--ref=" + ref,
                    "--in=" + in_file,
                    "--out=" + out_file,
                    "--warp=" + warp,
                    "--interp=trilinear",
                ]
            )
        )

    # If only a affmat-file has been specified perform affine registration to
    # MNI
    elif affmat and not warp:
        os.system(
            " ".join(
                [
                    op.join(fsl_dir, "flirt"),
                    "-ref " + ref,
                    "-in " + in_file,
                    "-out " + out_file,
                    "-applyxfm -init " + affmat,
                    "-interp trilinear",
                ]
            )
        )

    # If both a affmat- and warp-file have been defined, apply the warping
    # accordingly
    else:
        os.system(
            " ".join(
                [
                    op.join(fsl_dir, "applywarp"),
                    "--ref=" + ref,
                    "--in=" + in_file,
                    "--out=" + out_file,
                    "--warp=" + warp,
                    "--premat=" + affmat,
                    "--interp=trilinear",
                ]
            )
        )


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
    proj = HYPERPLANE[0] + np.dot(x.T, HYPERPLANE[1:])

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
    motion_ICs = features_df[
        "classification", features_df["classification"] == "rejected"
    ].index.values
    with open(op.join(out_dir, "classified_motion_ICs.txt"), "w") as fo:
        out_str = ",".join(motion_ICs.astype(str))
        fo.write(out_str)

    # Create a summary overview of the classification
    features_df.to_csv(
        op.join(out_dir, "classification_overview.txt"), sep="\t", index_label="IC"
    )

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
        print(
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
