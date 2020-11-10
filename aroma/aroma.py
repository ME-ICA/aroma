"""The core workflow for AROMA."""
import os
import os.path as op
import shutil

import nibabel as nib
import pandas as pd

from . import utils, features


def aroma_workflow(
    out_dir,
    in_feat=None,
    in_file=None,
    mc=None,
    mel_dir=None,
    affmat=None,
    warp=None,
    dim=0,
    den_type="nonaggr",
    mask=None,
    TR=None,
    overwrite=False,
    generate_plots=True,
):
    """Run the AROMA workflow.

    Parameters
    ----------
    in_feat
    """
    print("\n------------------------ RUNNING ICA-AROMA ------------------------")
    print("-------- 'ICA-based Automatic Removal Of Motion Artifacts' --------\n")
    if in_feat and in_file:
        raise ValueError("Only one of 'in_feat' and 'in_file' may be provided.")

    if in_feat and (mc or affmat or warp or mask):
        raise ValueError(
            "Arguments 'mc', 'affmat', 'warp', and 'mask' are incompatible "
            "with argument 'in_feat'."
        )

    # Define variables based on the type of input (i.e. Feat directory or
    # specific input arguments), and check whether the specified files exist.
    if in_feat:
        # Check whether the Feat directory exists
        if not op.isdir(in_feat):
            raise Exception("The specified FEAT directory does not exist.")

        # Define the variables which should be located in the Feat directory
        in_file = op.join(in_feat, "filtered_func_data.nii.gz")
        mc = op.join(in_feat, "mc", "prefiltered_func_data_mcf.par")
        affmat = op.join(in_feat, "reg", "example_func2highres.mat")
        warp = op.join(in_feat, "reg", "highres2standard_warp.nii.gz")

        # Check whether these files actually exist
        if not op.isfile(in_file):
            raise Exception("Missing filtered_func_data.nii.gz in Feat directory.")

        if not op.isfile(mc):
            raise Exception(
                "Missing mc/prefiltered_func_data_mcf.mat in Feat directory."
            )

        if not op.isfile(affmat):
            raise Exception("Missing reg/example_func2highres.mat in Feat directory.")

        if not op.isfile(warp):
            raise Exception(
                "Missing reg/highres2standard_warp.nii.gz in Feat directory."
            )

        # Check whether a melodic.ica directory exists
        if op.isdir(op.join(in_feat, "filtered_func_data.ica")):
            mel_dir = op.join(in_feat, "filtered_func_data.ica")
    else:
        # Check whether the files exist
        if not in_file:
            print("No input file specified.")
        elif not op.isfile(in_file):
            raise Exception("The specified input file does not exist.")

        if not mc:
            print("No mc file specified.")
        elif not op.isfile(mc):
            raise Exception("The specified mc file does does not exist.")

        if affmat and not op.isfile(affmat):
            raise Exception("The specified affmat file does not exist.")

        if warp and not op.isfile(warp):
            raise Exception("The specified warp file does not exist.")

    # Check if the mask exists, when specified.
    if mask and not op.isfile(mask):
        raise Exception("The specified mask does not exist.")

    # Check if the type of denoising is correctly specified, when specified
    if den_type not in ("nonaggr", "aggr", "both", "no"):
        print(
            "Type of denoising was not correctly specified. Non-aggressive "
            "denoising will be run."
        )
        den_type = "nonaggr"

    # Prepare

    # Define the FSL-bin directory
    fsl_dir = op.join(os.environ["FSLDIR"], "bin", "")

    # Create output directory if needed
    if op.isdir(out_dir) and not overwrite:
        print(
            "Output directory",
            out_dir,
            """already exists.
            AROMA will not continue.
            Rerun with the -overwrite option to explicitly overwrite
            existing output.""",
        )
        return
    elif op.isdir(out_dir) and overwrite:
        print(
            "Warning! Output directory {} exists and will be overwritten."
            "\n".format(out_dir)
        )
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    else:
        os.makedirs(out_dir)

    # Get TR of the fMRI data, if not specified
    if not TR:
        in_img = nib.load(in_file)
        TR = in_img.header.get_zooms()[3]

    # Check TR
    if TR == 1:
        print(
            "Warning! Please check whether the determined TR (of "
            + str(TR)
            + "s) is correct!\n"
        )
    elif TR == 0:
        raise Exception(
            "TR is zero. ICA-AROMA requires a valid TR and will therefore "
            "exit. Please check the header, or define the TR as an additional "
            "argument.\n"
            "-------------- ICA-AROMA IS CANCELED ------------\n"
        )

    # Define/create mask. Either by making a copy of the specified mask, or by
    # creating a new one.
    new_mask = op.join(out_dir, "mask.nii.gz")
    if mask:
        shutil.copyfile(mask, new_mask)
    elif in_feat and op.isfile(op.join(in_feat, "example_func.nii.gz")):
        # If a Feat directory is specified, and an example_func is present use
        # example_func to create a mask
        bet_command = "{0} {1} {2} -f 0.3 -n -m -R".format(
            op.join(fsl_dir, "bet"),
            op.join(in_feat, "example_func.nii.gz"),
            op.join(out_dir, "bet"),
        )
        os.system(bet_command)
        os.rename(op.join(out_dir, "bet_mask.nii.gz"), new_mask)
        if op.isfile(op.join(out_dir, "bet.nii.gz")):
            os.remove(op.join(out_dir, "bet.nii.gz"))
    else:
        if in_feat:
            print(
                " - No example_func was found in the Feat directory. "
                "A mask will be created including all voxels with varying "
                "intensity over time in the fMRI data. Please check!\n"
            )
        math_command = "{0} {1} -Tstd -bin {2}".format(
            op.join(fsl_dir, "fslmaths"), in_file, new_mask
        )
        os.system(math_command)

    # Run ICA-AROMA
    print("Step 1) MELODIC")
    utils.runICA(fsl_dir, in_file, out_dir, mel_dir, new_mask, dim, TR)

    print("Step 2) Automatic classification of the components")
    print("  - registering the spatial maps to MNI")
    mel_IC = op.join(out_dir, "melodic_IC_thr.nii.gz")
    mel_IC_MNI = op.join(out_dir, "melodic_IC_thr_MNI2mm.nii.gz")
    utils.register2MNI(fsl_dir, mel_IC, mel_IC_MNI, affmat, warp)

    print("  - extracting the CSF & Edge fraction features")
    features_df = pd.DataFrame()
    features_df["edge_fract"], features_df["csf_fract"] = features.feature_spatial(
        mel_IC_MNI
    )

    print("  - extracting the Maximum RP correlation feature")
    mel_mix = op.join(out_dir, "melodic.ica", "melodic_mix")
    features_df["max_RP_corr"] = features.feature_time_series(mel_mix, mc)

    print("  - extracting the High-frequency content feature")
    mel_FT_mix = op.join(out_dir, "melodic.ica", "melodic_FTmix")
    features_df["HFC"] = features.feature_frequency(mel_FT_mix, TR)

    print("  - classification")
    motion_ICs = utils.classification(features_df, out_dir)

    if generate_plots:
        from . import plotting

        plotting.classification_plot(
            op.join(out_dir, "classification_overview.tsv"), out_dir
        )

    if den_type != "no":
        print("Step 3) Data denoising")
        utils.denoising(fsl_dir, in_file, out_dir, mel_mix, den_type, motion_ICs)

    print("Finished")
