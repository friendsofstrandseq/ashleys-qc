import pandas as pd

from ashleyslib import AshleysEnum
from ashleyslib import Units
from ashleyslib import Columns
from ashleyslib import GenomicRegion


class FeatureMappingStatistics(AshleysEnum):
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


FTMS = FeatureMappingStatistics


def filter_read_alignments(count_statistics, min_mapq, read):
    """
    Categories in this filter function follow original
    "read count implementation" in MosaiCatcher.
    Note that statistics are update in-place and filter
    only returns start position of good reads
    """
    read_position = None
    if read.is_unmapped:
        count_statistics[FTMS.unmapped] += 1
    else:
        count_statistics[FTMS.mapped] += 1
        if read.is_supplementary or read.is_secondary or read.is_qcfail:
            count_statistics[FTMS.supplementary] += 1
        elif read.is_duplicate:
            count_statistics[FTMS.duplicate] += 1
        elif read.mapping_quality < min_mapq:
            count_statistics[FTMS.mapq] += 1
        elif read.is_read2:
            count_statistics[FTMS.read2] += 1
        else:
            count_statistics[FTMS.good] += 1
            read_position = read.reference_start
            if read.is_reverse:
                read_position *= -1
    return read_position


def finalize_mapstats_per_chromosome(library_name, chromosome, mapping_stats):
    """
    adds total read count and normalizes by total read count per chromosome
    """
    # add total read count for this chromosome
    mapping_stats[FTMS.total] = mapping_stats[FTMS.unmapped] + mapping_stats[FTMS.mapped]
    # turn into pd.Series
    mapping_stats = pd.Series(mapping_stats, dtype="int32")
    # normalize by total read count
    mapping_stats_relative = mapping_stats / mapping_stats[FTMS.total]
    mapping_stats = pd.DataFrame(
        [mapping_stats, mapping_stats_relative],
        index=pd.MultiIndex.from_tuples(
            [
                (library_name, chromosome, Units.absolute),
                (library_name, chromosome, Units.relative),
            ],
            names=[Columns.library, Columns.region, Columns.unit],
        ),
    )
    return mapping_stats


def aggregate_mapstats_genome_wide(feature_set, num_chromosomes):
    """ """
    libraries = feature_set.index.unique(level=Columns.library)

    wg_aggregates = []

    for lib in libraries:
        counts = feature_set.xs([lib, Units.absolute], level=[Columns.library, Columns.unit])
        assert counts.shape[0] == num_chromosomes
        counts = counts.sum(axis=0)
        norm_subset = counts / counts[FTMS.total]
        df = pd.DataFrame(
            [counts, norm_subset],
            columns=counts.index,
            index=pd.MultiIndex.from_tuples(
                [
                    (lib, GenomicRegion.wg, Units.absolute),
                    (lib, GenomicRegion.wg, Units.relative),
                ],
                names=[Columns.library, Columns.region, Columns.unit],
            ),
        )
        wg_aggregates.append(df)  # one DF per library
    wg_aggregates = pd.concat(wg_aggregates, axis=0, ignore_index=False)
    packed_table = pack_mapstats_table(wg_aggregates.copy())
    return wg_aggregates, packed_table


def pack_mapstats_table(wg_subset):
    """
    Prefix columns with window size
    """
    # since the mapping statistics are not window-based,
    # there is not much to do here except for selecting
    # the correct set of values
    wg_subset = wg_subset.xs(
        [GenomicRegion.wg, Units.relative], level=[Columns.region, Columns.unit]
    ).copy()
    # NB: important to copy aggregates b/c of drop
    # the relative value of the total reads is always 1
    wg_subset.drop(FTMS.total, axis=1, inplace=True)
    # the remaining level in the index is now only the library
    return wg_subset
