#!/usr/bin/env python

import sys
import argparse
import logging
import traceback

from ashleyslib import __version__
from ashleyslib import LOG_MESSAGE_FORMAT as logging_format

from ashleyslib.train_classification_model import add_training_parser
from ashleyslib.features import add_features_parser
from ashleyslib.prediction import add_prediction_parser
from ashleyslib.plotting import add_plotting_parser


def parse_command_line():
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument("--version", "-v", action="version", version=__version__)

    noise_level = parser.add_mutually_exclusive_group(required=False)
    noise_level.add_argument(
        "--debug",
        "-dbg",
        action="store_true",
        default=False,
        help="Print debug log messages to stderr.",
    )
    noise_level.add_argument(
        "--verbose",
        "-vrb",
        action="store_true",
        default=False,
        help="Print progress messages to stdout.",
    )

    parser.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of CPU cores to use. Default: 1 (no sanity checks!)",
    )

    parser.add_argument("--logging", "-l", default=None, dest="logging", help=argparse.SUPPRESS)

    parser = add_sub_parsers(parser)

    return parser.parse_args()


def add_sub_parsers(main_parser):
    subparsers = main_parser.add_subparsers(dest="subparser_name", title="Run modes")
    subparsers = add_features_parser(subparsers)
    subparsers = add_training_parser(subparsers)
    subparsers = add_prediction_parser(subparsers)
    subparsers = add_plotting_parser(subparsers)
    return main_parser


if __name__ == "__main__":
    args = parse_command_line()
    if args.debug:
        logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format=logging_format)
    elif args.verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=logging_format)
    else:
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format=logging_format)
    logger = logging.getLogger(None)
    logger.info("Logging system initialized")
    try:
        args.execute(args)
    except Exception:
        traceback.print_exc()
        raise
