#!/usr/bin/env python

import argparse
import sys

from ashleyslib.train_classification_model import add_training_parser
from ashleyslib.feature_generation import add_features_parser
from ashleyslib.prediction import add_prediction_parser


def parse_command_line():
    """
    :return:
    """
    parser = argparse.ArgumentParser(add_help=True)
    #parser.add_argument('--version', '-v', action='version', version=__version__)

    #parser_group = parser.add_argument_group('General parameters')
    #parser_group.add_argument('--jobs', '-j', type=int, default=1, dest='workers',
    #                     help='Number of CPU cores to use, default: 1')

    parser = add_sub_parsers(parser)

    return parser.parse_args()


def add_sub_parsers(main_parser):
    """
    :param main_parser:
    :return:
    """
    subparsers = main_parser.add_subparsers(dest='subparser_name', title='Run modes')
    subparsers = add_features_parser(subparsers)
    subparsers = add_training_parser(subparsers)
    subparsers = add_prediction_parser(subparsers)
    return main_parser


if __name__ == '__main__':
    args = parse_command_line()
    args.execute(args)
