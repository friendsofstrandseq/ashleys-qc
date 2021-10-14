import numpy as np
import pandas as pd

from ashleyslib import AshleysEnum
from ashleyslib import Columns
from ashleyslib import Units
from ashleyslib import GenomicRegion


class FeatureWindowCounts(AshleysEnum):
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

    @classmethod
    def get_window_labels(cls, as_string=False):
        labels = [cls(i) for i in range(10, 110, 10)]
        if as_string:
            labels = [str(label) for label in labels]
        return labels

    @classmethod
    def get_extended_labels(cls, as_string=False):
        labels = [cls(i) for i in range(10, 110, 10)]
        labels += [cls.NZERO, cls.TOTAL]
        if as_string:
            labels = [str(label) for label in labels]
        return labels


FTWCNT = FeatureWindowCounts


def collect_window_count_statistics_per_chromosome(
    crick_reads, watson_reads, window_sizes, chromosome, chrom_size, library_name
):
    """
    Main function to compute crick/watson read count-based features;
    The features are returned in "Watson window" binned form:
    W10, W20, ... W100
    """
    collect_window_statistics = []
    window_statistics_index = []

    # generic bins [0...10...20.......100]
    # to count number of windows having
    # <10, <20, <30 ... percent of Watson reads
    decile_bins = np.arange(0, 101, 10)
    decile_labels = FTWCNT.get_window_labels()
    assert len(decile_labels) == 10

    for window_size in window_sizes:
        # number of windows needs to be
        # recorded for data normalization
        num_windows = 0
        num_nonzero_windows = 0

        watson_decile_bins_abs = np.zeros(len(decile_labels), dtype=np.int32)
        for step in [0, window_size // 2]:
            # adjust chromosome size to include last incomplete
            # window; for larger window sizes such as 5 Mbp, this
            # can be a considerable fraction of the chromosome
            adjusted_chrom_size = (chrom_size + step) // window_size * window_size + window_size

            # +1 here to have last window created
            windows = np.arange(step, adjusted_chrom_size + 1, window_size)

            # check that start of last window is always smaller than
            # end of current chromosome
            assert (
                windows[-2] < chrom_size
            ), f"{chromosome} invalid window start: {windows[-2:]} >= {chrom_size} [{adjusted_chrom_size} / {window_size} / {step}]"
            # check that end of last window is always overlapping
            # end of current chromosome
            assert (
                windows[-1] >= chrom_size
            ), f"{chromosome} invalid window end: {windows[-2:]} < {chrom_size} [{adjusted_chrom_size} / {window_size} / {step}]"

            # windows.size is length of list, but we need
            # number of windows (bins) here, i.e. -1
            num_windows += windows.size - 1
            crick_counts, _ = np.histogram(crick_reads, windows)
            watson_counts, _ = np.histogram(watson_reads, windows)
            total_counts = crick_counts + watson_counts
            # avoid division by zero
            nz_idx = total_counts > 0
            num_nonzero_windows += nz_idx.sum()
            watson_pct_per_window = watson_counts[nz_idx] / total_counts[nz_idx] * 100
            watson_decile_counts, _ = np.histogram(watson_pct_per_window, decile_bins)
            # sum up decile counts
            watson_decile_bins_abs += watson_decile_counts

        # append window and non-zero window count
        extended_decile_bins = np.append(
            watson_decile_bins_abs, [num_nonzero_windows, num_windows], axis=0
        )
        # turn into Pandas Series
        counts = pd.Series(
            extended_decile_bins, index=FTWCNT.get_extended_labels(), dtype=np.int32
        )
        # store absolute counts (not read counts);
        # these are Watson window counts
        collect_window_statistics.append(counts)
        window_statistics_index.append((library_name, chromosome, window_size, Units.absolute))
        # normalize Watson window counts by non-zero windows
        relative = counts / counts[FTWCNT.nzero]
        # normalize non-zero windows by total windows
        relative[[FTWCNT.nzero, FTWCNT.total]] = (
            counts[[FTWCNT.nzero, FTWCNT.total]] / counts[FTWCNT.total]
        )
        collect_window_statistics.append(relative)
        window_statistics_index.append((library_name, chromosome, window_size, Units.relative))

    collect_window_statistics = pd.concat(collect_window_statistics, axis=1, ignore_index=False)
    collect_window_statistics = collect_window_statistics.transpose()
    collect_window_statistics.index = pd.MultiIndex.from_tuples(
        window_statistics_index,
        names=[Columns.library, Columns.region, Columns.window_size, Columns.unit],
    )
    return collect_window_statistics


def aggregate_window_count_features(feature_set, num_chromosomes):
    """
    Aggregate window count features per library and window size,
    and merge all aggregated records into one DF
    """
    libraries = feature_set.index.unique(level=Columns.library)
    window_sizes = feature_set.index.unique(level=Columns.window_size)

    wg_aggregates = []

    for lib in libraries:
        for window in window_sizes:
            counts = feature_set.xs(
                [lib, window, Units.absolute],
                level=[Columns.library, Columns.window_size, Columns.unit],
            )
            assert counts.shape[0] == num_chromosomes
            counts = counts.sum(axis=0)
            norm_subset = counts / counts[FTWCNT.nzero]
            norm_subset[[FTWCNT.nzero, FTWCNT.total]] = (
                counts[[FTWCNT.nzero, FTWCNT.total]] / counts[FTWCNT.total]
            )
            df = pd.DataFrame(
                [counts, norm_subset],
                columns=counts.index,
                index=pd.MultiIndex.from_tuples(
                    [
                        (lib, GenomicRegion.wg, window, Units.absolute),
                        (lib, GenomicRegion.wg, window, Units.relative),
                    ],
                    names=[Columns.library, Columns.region, Columns.window_size, Columns.unit],
                ),
            )
            wg_aggregates.append(df)  # one DF per library and window size
    wg_aggregates = pd.concat(wg_aggregates, axis=0, ignore_index=False)
    packed_table = pack_wincount_table(wg_aggregates)
    return wg_aggregates, packed_table


def pack_wincount_table(wg_subset):
    """ """

    window_subsets = []
    for window_size in wg_subset.index.unique(level=Columns.window_size):
        window_subset = wg_subset.xs(
            [GenomicRegion.wg, window_size, Units.relative],
            level=[Columns.region, Columns.window_size, Columns.unit],
        ).copy()
        window_subset.drop(FTWCNT.total, axis=1, inplace=True)
        window_subset.columns = [f"WS{window_size}_{c}" for c in window_subset.columns]
        window_subsets.append(window_subset)

    window_subsets = pd.concat(window_subsets, axis=1, ignore_index=False)
    window_subsets.sort_index(inplace=True)
    return window_subsets
