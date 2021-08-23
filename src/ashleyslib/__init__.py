# coding=utf-8

import enum

# package info
__author__ = "Christina Eimer"
__email__ = "eimer@mpi-inf.mpg.de"
__version__ = "0.2.0dev"
__contributor__ = "Peter Ebert"
__email_contributor__ = "peter.ebert@iscb.org"

LOG_MESSAGE_FORMAT = "%(asctime)s | %(funcName)s | %(levelname)s | %(message)s"


class Scales(enum.Enum):
    ABSOLUTE = 1
    absolute = 1
    RELATIVE = 2
    relative = 2

    def __str__(self):
        return self.name


class FeatureMappingStatistics(enum.Enum):
    UNMAPPED = 1
    MAPPED = 2
    SUPPLEMENTARY = 3
    DUPLICATE = 4
    MAPQ = 5
    READ2 = 6
    GOOD = 7
    TOTAL = 8

    unmapped = 1
    mapped = 2
    supplementary = 3
    duplicate = 4
    mapq = 5
    read2 = 6
    good = 7
    total = 8

    def __str__(self):
        return self.name


class FeatureWindowStatistics(enum.Enum):
    W10 = 10
    W20 = 20
    W30 = 30
    W40 = 40
    W50 = 50
    W60 = 60
    W70 = 70
    W80 = 80
    W90 = 90
    W100 = 100

    TOTAL = 1000
    NZERO = 2000
    NEMPTY = 2000

    total = 1000
    nzero = 2000
    nempty = 2000

    def __str__(self):
        return self.name

    @classmethod
    def get_window_labels(cls):
        labels = [f"W{i}" for i in range(10, 101, 10)]
        return labels

    @classmethod
    def get_extended_labels(cls):
        labels = [f"W{i}" for i in range(10, 101, 10)]
        labels += ["NZERO", "TOTAL"]
        return labels


class ReadCounts(enum.Enum):
    CRICK = 1
    WATSON = 2
    TOTAL = 3

    crick = 1
    watson = 2
    total = 3

    plus = 1
    minus = 2

    forward = 1
    reverse = 2

    def __str__(self):
        return self.name
