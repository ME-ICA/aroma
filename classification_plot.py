#!/usr/bin/env python
"""Parser for classification plot-generation workflow."""
import argparse

from aroma import plotting


def _get_parser():
    parser = argparse.ArgumentParser(
        description=("Plot component classification overview similar to plot "
                     "in the main AROMA paper")
    )
    # Required options
    reqoptions = parser.add_argument_group('Required arguments')
    reqoptions.add_argument('-i', '-in',
                            dest='myinput',
                            required=True,
                            help="""Input query or filename.
                                    Use quotes when specifying a query""")

    optoptions = parser.add_argument_group('Optional arguments')
    optoptions.add_argument('-out_dir',
                            dest='out_dir',
                            required=False,
                            default='.',
                            help="""Specification of directory
                                    where figure will be saved""")
    optoptions.add_argument('-type',
                            dest='plottype',
                            required=False,
                            default='assessment',
                            help="""Specification of the type of plot you want.
                                    Currently this is a placeholder option for
                                    potential other plots that might be added
                                    in the future.""")
    return parser


def _main(argv=None):
    """Entry point for classification_plot CLI."""
    options = _get_parser().parse_args(argv)
    if options.plottype == 'assessment':
        plotting.classification_plot(options.myinput, options.out_dir)


if __name__ == "__main__":
    _main()
