#!/usr/bin/env python
"""Parser for classification plot-generation workflow."""
import argparse

from aroma import plotting
from aroma.cli.parser_utils import is_valid_file


def _get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot component classification overview similar to plot "
            "in the main AROMA paper"
        )
    )
    parser.add_argument(
        dest="in_file",
        type=lambda x: is_valid_file(parser, x),
        help="Tab-delimited file with component classifications.",
    )
    parser.add_argument(
        "-out_dir",
        dest="out_dir",
        required=False,
        type=str,
        default=".",
        help="Specification of directory where figure will be saved.",
    )
    return parser


def _main(argv=None):
    """Entry point for classification_plot CLI."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    plotting.classification_plot(**kwargs)


if __name__ == "__main__":
    _main()
