# coding=utf-8

import enum

# package info
__author__ = "Christina Eimer"
__email__ = "eimer@mpi-inf.mpg.de"
__version__ = "0.2.0dev"
__contributor__ = "Peter Ebert"
__email_contributor__ = "peter.ebert@iscb.org"

LOG_MESSAGE_FORMAT = "%(asctime)s | %(funcName)s | %(levelname)s | %(message)s"


class AshleysEnum(enum.Enum):
    def __str__(self):
        return self.name

    def __name__(self):
        return self.name


class FeatureTypes(AshleysEnum):
    mapping_statistics = 1
    mapping_stats = 1
    mapping = 1
    mapstats = 1
    mapstat = 1

    window_counts = 2
    window_count = 2
    windows_counts = 2
    wincount = 2
    wincounts = 2
    wc_count = 2
    watson_crick = 2
    wc = 2
    wc_stats = 2
    wc_statistics = 2

    window_divergence = 3
    windows_divergence = 3
    windiv = 3
    window_div = 3
    window_divergences = 3


class Orientation(AshleysEnum):
    crick = 1
    watson = 2
    C = 1
    W = 2
    c = 1
    w = 2

    plus = 1
    minus = 2

    forward = 1
    reverse = 2


class Columns(AshleysEnum):
    library = 1
    lib_id = 1
    library_id = 1
    library_name = 1
    cell = 1
    cell_id = 1
    cell_name = 1

    region = 2

    window_size = 3
    window = 3
    ws = 3

    unit = 4
    units = 4

    sample = 5
    sample_id = 5
    sample_name = 5

    subset = 6

    @classmethod
    def get_index_columns(cls):
        idx_columns = [
            cls["library"],
            cls["region"],
            cls["window_size"],
            cls["unit"],
        ]
        return idx_columns

    @classmethod
    def is_index_column(cls, column_name):
        if isinstance(column_name, str):
            column = Columns[column_name]
        elif isinstance(column_name, int):
            column = Columns(column_name)
        elif isinstance(column_name, Columns):
            column = column_name
        else:
            raise ValueError(
                "Unsupported type to create Enum(Columns): "
                f"{type(column_name)} / {column_name}. Must be "
                "int, string or Columns."
            )
        return column in cls.get_index_columns()


class Units(AshleysEnum):
    ABSOLUTE = 1
    absolute = 1
    RELATIVE = 2
    relative = 2
    BITS = 3
    bits = 3
    BIT = 3
    bit = 3


class GenomicRegion(AshleysEnum):
    wg = 1
    whole_genome = 1
    wholegenome = 1
    genome_wide = 1
    genomewide = 1
    genome = 1

    # TODO this is only needed for windiv
    # features at the moment, and quite an
    # ugly place here...
    complete = 2
    cmp = 2

    quartile = 3
    qrt = 3
