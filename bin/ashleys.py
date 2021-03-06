#!/usr/bin/env python

import argparse

from ashleyslib.train_classification_model import add_training_parser
from ashleyslib.feature_generation import add_features_parser
from ashleyslib.prediction import add_prediction_parser
from ashleyslib.plotting import add_plotting_parser


def parse_command_line():
    parser = argparse.ArgumentParser(add_help=True)
    # parser.add_argument('--version', '-v', action='version', version=__version__)

    parser_group = parser.add_argument_group('General parameters')
    parser_group.add_argument('--jobs', '-j', type=int, default=1, help='Number of CPU cores to use, default: 1')
    parser_group.add_argument('--logging', '-l', help='file name for logging output')

    parser = add_sub_parsers(parser)

    return parser.parse_args()


def add_sub_parsers(main_parser):
    subparsers = main_parser.add_subparsers(dest='subparser_name', title='Run modes')
    subparsers = add_features_parser(subparsers)
    subparsers = add_training_parser(subparsers)
    subparsers = add_prediction_parser(subparsers)
    subparsers = add_plotting_parser(subparsers)
    return main_parser


if __name__ == '__main__':
    args = parse_command_line()
    args.execute(args)
