#!/usr/bin/env python
"""Parser for AROMA workflow."""
import argparse

from aroma import aroma
from aroma.cli.parser_utils import is_valid_file


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
        "-i",
        "-in",
        dest="in_file",
        type=lambda x: is_valid_file(parser, x),
        required=True,
        help="Input file name of fMRI data (.nii.gz) in MNI space",
    )
    reqoptions.add_argument(
        "--mixing",
        dest="mixing",
        type=lambda x: is_valid_file(parser, x),
        required=True,
        help="Mixing matrix from ICA",
    )
    reqoptions.add_argument(
        "-mc",
        dest="mc",
        type=lambda x: is_valid_file(parser, x),
        required=True,
        help=(
            "File name of the motion parameters obtained after motion "
            "realignment (e.g., FSL MCFLIRT). Note that the order of "
            "parameters does not matter, should your file not originate "
            "from FSL MCFLIRT."
        ),
    )
    reqoptions.add_argument(
        "--components",
        dest="component_maps",
        type=lambda x: is_valid_file(parser, x),
        required=True,
        help=(
            "Z-statistic component maps, thresholded with mixture modeling at >0.5, in MNI space."
        ),
    )
    reqoptions.add_argument(
        "-o",
        "-out",
        dest="out_dir",
        required=True,
        help="Output directory name",
    )

    # Optional options
    optoptions = parser.add_argument_group("Optional arguments")
    optoptions.add_argument(
        "-mcsource",
        dest="mc_source",
        choices=["auto", "fsl", "fmriprep", "spm", "afni"],
        required=False,
        default="auto",
        help=(
            "Source (and format) of motion parameters. "
            "Each package saves its motion parameters slightly differently, "
            "so we need to determine the source before using the parameters "
            "in AROMA. "
            "The 'auto' option attempts to predict the source of the "
            "parameters based on the filename. "
            "Default is 'auto'."
        ),
    )
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
