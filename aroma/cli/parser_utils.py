"""Utility functions for CLI parsers."""
import os.path as op


def is_valid_file(parser, arg):
    """Check if argument is existing file.

    Parameters
    ----------
    parser : argparse.Parser
        Parser from which arg comes.
    arg : str
        Argument value from parser.

    Returns
    -------
    arg : str
        Input argument, *if* arg is a file.
    """
    if not op.isfile(arg) and arg is not None:
        parser.error('The file {0} does not exist!'.format(arg))

    return arg


def is_valid_path(parser, arg):
    """Check if argument is existing directory.

    Parameters
    ----------
    parser : argparse.Parser
        Parser from which arg comes.
    arg : str
        Argument value from parser.

    Returns
    -------
    arg : str
        Input argument, *if* arg is a folder.
    """
    if not op.isdir(arg) and arg is not None:
        parser.error('The folder {0} does not exist!'.format(arg))

    return arg
