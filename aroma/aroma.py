"""The core workflow for AROMA."""
import datetime
import logging
import os
import os.path as op
import shutil

import nibabel as nib
import pandas as pd

from aroma import _version, features, utils

LGR = logging.getLogger(__name__)


def aroma_workflow(
    in_file,
    mc,
    mixing,
    component_maps,
    mask,
    out_dir,
    den_type="nonaggr",
    TR=None,
    overwrite=False,
    generate_plots=True,
    debug=False,
    quiet=False,
    mc_source="auto",
):
    """Run the AROMA workflow.

    Parameters
    ----------
    in_file : str
        Path to MNI-space functional run to denoise.
    mc : str
        Path to motion parameters.
    mixing : str
        Path to mixing matrix.
    component_maps : str
        Path to thresholded z-statistic component maps.
    mask : str
        Path to binary brain mask, in MNI space.
    out_dir : str
        Output directory.
    den_type : {"nonaggr", "aggr", "both", "no"}, optional
        Denoising approach to use.
    TR : float or None, optional
        Repetition time of data in in_file and mixing.
        If None, this will be extracted from the header of in_file.
    overwrite : bool
    generate_plots : bool
    debug : bool
    quiet : bool
    mc_source : {"auto"}, optional
        What format is the mc file in?
    """
    assert op.isfile(in_file)
    assert op.isfile(mc)
    assert op.isfile(mixing)
    assert op.isfile(component_maps)

    # Create output directory if needed
    if op.isdir(out_dir) and not overwrite:
        LGR.info(
            f"Output directory {out_dir},"
            """already exists.
            AROMA will not continue.
            Rerun with the -overwrite option to explicitly overwrite
            existing output.""",
        )
        return
    elif op.isdir(out_dir) and overwrite:
        LGR.warning(
            "Output directory {} exists and will be overwritten."
            "\n".format(out_dir)
        )
        shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    else:
        os.makedirs(out_dir)

    # Create logfile name
    basename = 'aroma_'
    extension = 'tsv'
    isotime = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
    logname = os.path.join(out_dir, (basename + isotime + '.' + extension))

    # Set logging format
    log_formatter = logging.Formatter(
        '%(asctime)s\t%(name)-12s\t%(levelname)-8s\t%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')

    # Set up logging file and open it for writing
    log_handler = logging.FileHandler(logname)
    log_handler.setFormatter(log_formatter)
    sh = logging.StreamHandler()

    # add logger mode options
    if quiet:
        logging.basicConfig(level=logging.WARNING,
                            handlers=[log_handler, sh],
                            format='%(levelname)-10s %(message)s')
    elif debug:
        logging.basicConfig(level=logging.DEBUG,
                            handlers=[log_handler, sh],
                            format='%(levelname)-10s %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                            handlers=[log_handler, sh],
                            format='%(levelname)-10s %(message)s')
    version_number = _version.get_versions()['version']
    LGR.info(f'Currently running ICA-AROMA version {version_number}')

    # Check if the mask exists, when specified.
    if mask and not op.isfile(mask):
        raise Exception("The specified mask does not exist.")

    # Check if the type of denoising is correctly specified, when specified
    if den_type not in ("nonaggr", "aggr", "both", "no"):
        LGR.warning(
            "Type of denoising was not correctly specified. Non-aggressive "
            "denoising will be run."
        )
        den_type = "nonaggr"

    # Prepare
    # Get TR of the fMRI data, if not specified
    if not TR:
        in_img = nib.load(in_file)
        TR = in_img.header.get_zooms()[3]

    # Check TR
    if TR == 1:
        LGR.warning(
            "Please check whether the determined TR (of "
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

    # Save a copy of the mask in the output directory
    new_mask = op.join(out_dir, "mask.nii.gz")
    shutil.copyfile(mask, new_mask)

    LGR.info("  - extracting the CSF & Edge fraction features")
    features_df = pd.DataFrame()
    features_df["edge_fract"], features_df["csf_fract"] = features.feature_spatial(
        component_maps
    )

    LGR.info("  - extracting the Maximum RP correlation feature")
    mc = utils.load_motpars(mc, source=mc_source)
    features_df["max_RP_corr"] = features.feature_time_series(mixing, mc)

    LGR.info("  - extracting the High-frequency content feature")
    mel_FT_mix = utils.calculate_ft(mixing, TR)
    features_df["HFC"] = features.feature_frequency(mel_FT_mix, TR)

    LGR.info("  - classification")
    motion_ICs = utils.classification(features_df, out_dir)

    if generate_plots:
        from . import plotting

        plotting.classification_plot(
            op.join(out_dir, "classification_overview.tsv"), out_dir
        )

    if den_type != "no":
        LGR.info("Step 3) Data denoising")
        utils.denoising(in_file, out_dir, mixing, den_type, motion_ICs)

    LGR.info("Finished")
