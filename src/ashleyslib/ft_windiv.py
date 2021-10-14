import operator as op

import numpy as np
import pandas as pd
import scipy.stats as stats

from ashleyslib import AshleysEnum, GenomicRegion, Units, Columns


class FeatureWindowDivergence(AshleysEnum):

    KLDCMP_MIN = 1
    KLDCMP_MAX = 2
    KLDCMP_MEAN = 3
    KLDCMP_MEDIAN = 4
    KLDCMP_STDDEV = 5

    KLDQRT_MIN = 6
    KLDQRT_MAX = 7
    KLDQRT_MEAN = 8
    KLDQRT_MEDIAN = 9
    KLDQRT_STDDEV = 10

    kldcmp_min = 1
    kldcmp_max = 2
    kldcmp_mean = 3
    kldcmp_median = 4
    kldcmp_stddev = 5

    kldqrt_min = 6
    kldqrt_max = 7
    kldqrt_mean = 8
    kldqrt_median = 9
    kldqrt_stddev = 10

    @classmethod
    def get_div_label(cls, which):
        if which.lower() in ["cmp", "complete"]:
            name = "KLDCMP"
        elif which.lower() in ["qrt", "quartile"]:
            name = "KLDQRT"
        else:
            raise ValueError(f"Unknown divergence label: {which}")
        return name

    @classmethod
    def get_feature_suffixes(cls):
        return ["MIN", "MAX", "MEAN", "MEDIAN", "STDDEV"]


FTWDIV = FeatureWindowDivergence


def collect_window_divergence_statistics_per_chromosome(
    crick_reads, watson_reads, window_sizes, chromosome, chrom_size, library_name
):

    win_div_statistics = []
    win_div_index = []

    for window_size in window_sizes:

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

            crick_counts, _ = np.histogram(crick_reads, windows)
            watson_counts, _ = np.histogram(watson_reads, windows)

            # add fudge factor of 1 to avoid dealing with log(0)
            crick_counts += 1
            watson_counts += 1

            # compute global and quartile KLD in both directions
            win_div_statistics.append(stats.entropy(crick_counts, watson_counts))
            win_div_statistics.append(stats.entropy(watson_counts, crick_counts))
            win_div_index.append(
                (library_name, chromosome, window_size, Units.bits, GenomicRegion.complete)
            )
            win_div_index.append(
                (library_name, chromosome, window_size, Units.bits, GenomicRegion.complete)
            )

            start_pos = 0
            for quartile in [0.25, 0.5, 0.75, 1.0]:
                end_pos = int(crick_counts.size * quartile)
                win_div_statistics.append(
                    stats.entropy(
                        crick_counts[start_pos:end_pos], watson_counts[start_pos:end_pos]
                    )
                )
                win_div_index.append(
                    (library_name, chromosome, window_size, Units.bits, GenomicRegion.quartile)
                )
                win_div_statistics.append(
                    stats.entropy(
                        watson_counts[start_pos:end_pos], crick_counts[start_pos:end_pos]
                    )
                )
                win_div_index.append(
                    (library_name, chromosome, window_size, Units.bits, GenomicRegion.quartile)
                )
                start_pos = end_pos

    win_div_index = pd.MultiIndex.from_tuples(
        win_div_index,
        names=[Columns.library, Columns.region, Columns.window_size, Columns.unit, Columns.subset],
    )
    win_div_statistics = pd.Series(win_div_statistics, index=win_div_index)

    return win_div_statistics


def aggregate_window_divergence_features(feature_set, num_chromosomes):
    """
    Aggregate window divergence features per library and window size,
    and merge all aggregated records into one DF
    """
    libraries = feature_set.index.unique(level=Columns.library)
    window_sizes = feature_set.index.unique(level=Columns.window_size)
    subsets = feature_set.index.unique(level=Columns.subset)

    stats_labels = ["min", "mean", "50%", "max", "std"]
    stats_suffix = FTWDIV.get_feature_suffixes()

    get_stats = op.itemgetter(*tuple(stats_labels))

    wg_aggregates = []

    for lib in libraries:
        for window in window_sizes:

            subset_stats = []
            subset_labels = []

            for subset in subsets:
                divergences = feature_set.xs(
                    [lib, window, Units.bits, subset],
                    level=[Columns.library, Columns.window_size, Columns.unit, Columns.subset],
                )
                assert divergences.shape[0] % num_chromosomes == 0
                stats = get_stats(divergences.describe())
                subset_stats.extend(stats)
                ftlabel = FTWDIV.get_div_label(str(subset))
                labels = [f"{ftlabel}_{suffix}" for suffix in stats_suffix]
                subset_labels.extend(labels)

            df = pd.DataFrame(
                [subset_stats],
                columns=subset_labels,
                index=pd.MultiIndex.from_tuples(
                    [
                        (lib, GenomicRegion.wg, window, Units.bits),
                    ],
                    names=[
                        Columns.library,
                        Columns.region,
                        Columns.window_size,
                        Columns.unit,
                    ],
                ),
            )
            wg_aggregates.append(df)  # one DF per library and window size
    wg_aggregates = pd.concat(wg_aggregates, axis=0, ignore_index=False)
    packed_table = pack_windiv_table(wg_aggregates)
    return wg_aggregates, packed_table


def pack_windiv_table(wg_subset):
    """ """
    window_subsets = []
    for window_size in wg_subset.index.unique(level=Columns.window_size):
        window_subset = wg_subset.xs(
            [GenomicRegion.wg, window_size, Units.bits],
            level=[Columns.region, Columns.window_size, Columns.unit],
        ).copy()
        window_subset.columns = [f"WS{window_size}_{c}" for c in window_subset.columns]
        window_subsets.append(window_subset)
    window_subsets = pd.concat(window_subsets, axis=1, ignore_index=False)
    window_subsets.sort_index(inplace=True)
    return window_subsets
