#!/usr/bin/env python
"""Parser for AROMA workflow."""
import argparse

from aroma import aroma
from aroma.cli.parser_utils import is_valid_file, is_valid_path


def _get_parser():
    """
    Parse command line inputs for aroma.

    Returns
    -------
    parser.parse_args() : argparse dict
    """
    parser = argparse.ArgumentParser(
        description=(
            "Script to run ICA-AROMA v0.3 beta ('ICA-based "
            "Automatic Removal Of Motion Artifacts') on fMRI data. "
            "See the companion manual for further information."
        )
    )

    # Required options
    reqoptions = parser.add_argument_group("Required arguments")
    reqoptions.add_argument(
        "-o", "-out",
        dest="out_dir",
        required=True,
        help="Output directory name"
    )

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "-i",
        "-in",
        dest="in_file",
        type=lambda x: is_valid_file(parser, x),
        required=False,
        help="Input file name of fMRI data (.nii.gz)",
    )
    inputs.add_argument(
        "-f",
        "-feat",
        dest="in_feat",
        required=False,
        type=lambda x: is_valid_path(parser, x),
        help=(
            "Feat directory name (Feat should have been run without temporal "
            "filtering and including registration to MNI152)"
        ),
    )

    # Required options in non-Feat mode
    nonfeatoptions = parser.add_argument_group(
        "Required arguments - generic mode",
        description=(
            "These arguments should only be provided if the primary input is "
            "an fMRI file, and not a FEAT directory."
        )
    )
    nonfeatoptions.add_argument(
        "-mc",
        dest="mc",
        type=lambda x: is_valid_file(parser, x),
        required=False,
        help=(
            "File name of the motion parameters obtained after motion "
            "realignment (e.g., FSL MCFLIRT). Note that the order of "
            "parameters does not matter, should your file not originate "
            "from FSL MCFLIRT."
        ),
    )
    nonfeatoptions.add_argument(
        "-a",
        "-affmat",
        dest="affmat",
        type=lambda x: is_valid_file(parser, x),
        default=None,
        help=(
            "File name of the mat-file describing the affine registration "
            "(e.g., FSL FLIRT) of the functional data to structural space "
            "(.mat file). (e.g., "
            "/home/user/PROJECT/SUBJECT.feat/reg/example_func2highres.mat"
        ),
    )
    nonfeatoptions.add_argument(
        "-w",
        "-warp",
        dest="warp",
        type=lambda x: is_valid_file(parser, x),
        default=None,
        help=(
            "File name of the warp-file describing the non-linear "
            "registration (e.g., FSL FNIRT) of the structural data to MNI152 "
            "space (.nii.gz). (e.g., "
            "/home/user/PROJECT/SUBJECT.feat/reg/highres2standard_warp.nii.gz"
        ),
    )
    nonfeatoptions.add_argument(
        "-m",
        "-mask",
        dest="mask",
        type=lambda x: is_valid_file(parser, x),
        default=None,
        help=(
            "File name of the mask to be used for MELODIC (denoising will be "
            "performed on the original/non-masked input data)"
        ),
    )

    # Optional options
    optoptions = parser.add_argument_group("Optional arguments")
    optoptions.add_argument("-tr", dest="TR", help="TR in seconds", type=float)
    optoptions.add_argument(
        "-den",
        dest="den_type",
        default="nonaggr",
        choices=["nonaggr", "aggr", "both", "no"],
        help=(
            "Type of denoising strategy: 'no': only classification, no "
            "denoising; 'nonaggr': non-aggresssive denoising (default); "
            "'aggr': aggressive denoising; 'both': both aggressive and "
            "non-aggressive denoising (seperately)"
        ),
    )
    optoptions.add_argument(
        "-md",
        "-mel_dir",
        dest="mel_dir",
        type=lambda x: is_valid_path(parser, x),
        default="",
        help=(
            "MELODIC directory name, in case MELODIC has been run previously."
        ),
    )
    optoptions.add_argument(
        "-dim",
        dest="dim",
        default=0,
        help=(
            "Dimensionality reduction into #num dimensions when running "
            "MELODIC (default: automatic estimation; i.e. -dim 0)"
        ),
        type=int,
    )
    optoptions.add_argument(
        "-ow",
        "-overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite existing output",
        default=False,
    )
    optoptions.add_argument(
        "-np",
        "-noplots",
        dest="generate_plots",
        action="store_false",
        help=(
            "Plot component classification overview similar to plot in the "
            "main AROMA paper"
        ),
        default=True,
    )

    return parser


def _main(argv=None):
    """Entry point for aroma CLI."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    aroma.aroma_workflow(**kwargs)


if __name__ == "__main__":
    _main()
