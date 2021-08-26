import os
import pathlib as pl
import random as rand
import multiprocessing as mp
import collections as col
import functools as fnt
import re
import logging

# change in the numexpr package / dep of pandas
# if not set, issues pointless info logging message
os.environ["NUMEXPR_MAX_THREADS"] = "2"

import pandas as pd
import numpy as np
import pysam

from ashleyslib import FeatureMappingStatistics as ftms
from ashleyslib import FeatureWindowStatistics as ftws
from ashleyslib import Scales as scales


logger = logging.getLogger(__name__)


DEFAULT_WINDOW_SIZES = ["5e6", "2e6", "1e6", "8e5", "6e5", "4e5", "2e5"]
DEFAULT_WINDOW_SIZES_STRING = " ".join(DEFAULT_WINDOW_SIZES)
# for the default values, need to cast to int manually
DEFAULT_WINDOW_SIZES = list(map(lambda x: int(float(x)), DEFAULT_WINDOW_SIZES))


# fmt: off
def add_features_parser(subparsers):

    parser = subparsers.add_parser(
        "features",
        help="Compute features for (indexed) Strand-seq BAM files"
    )

    io_group = parser.add_argument_group('Input/output parameters')
    io_group.add_argument(
        "--input-bam",
        "-i",
        nargs="+",
        type=lambda x: pl.Path(x).resolve().absolute(),
        required=True,
        dest="input_bam",
        metavar="INPUT_BAM",
        help="Single string or list of strings to be interpreted as (i) "
                "path(s) to individual BAM file(s), (ii) path(s) to folder(s) "
                "containing BAM files (potentially collected recursively "
                "[see --recursive]), or (iii) a mix of the "
                "above. In all cases, the BAM files have to be indexed "
                "and the index file must have the same name with file "
                "extension .bam.bai"
    )
    io_group.add_argument(
        "--bam-extension",
        "-e",
        type=str,
        dest="bam_extension",
        default=".bam",
        help="File extension of BAM files for input file collection. "
             "Default: .bam"
    )
    io_group.add_argument(
        "--recursive",
        "-r",
        dest="recursive",
        action="store_true",
        default=False,
        help="If set, input file collection will recurse into subfolders. "
             "Default: False"
    )
    io_group.add_argument(
        "--chromosomes",
        "-c",
        type=str,
        dest='chromosomes',
        default="^(chr)?[0-9X]+$",
        help="Specify regular expression to select chromosomes for "
             "feature computation. Default: (chr)1-22, X",
    )
    io_group.add_argument(
        "--discard-read-positions",
        "-d",
        action="store_true",
        default=False,
        dest="discard_read_positions",
        help="Do not save/store alignment positions of (good) reads. "
             "Default: False"
    )
    io_group.add_argument(
        "--output-features",
        "-o",
        type=lambda p: pl.Path(p).resolve().absolute(),
        required=True,
        dest="output_features",
        help="Path to output file (TSV feature table). Non-existing "
             "folders will be created."
    )
    io_group.add_argument(
        "--output-plotting",
        "-p",
        default=None,
        dest="output_plotting",
        help="Path to output file (TSV plotting table). Non-existing "
             "folders will be created."
    )

    ft_group = parser.add_argument_group("Feature generation parameters")
    ft_group.add_argument(
        "--window-size",
        "-w",
        type=lambda x: int(float(x)),
        nargs="+",
        dest="window_size",
        default=DEFAULT_WINDOW_SIZES,
        help="Window size(s) for feature generation (integers "
             "or list of integers). For convenience, scientific "
             "(e) notation can be used. Default: " +
             DEFAULT_WINDOW_SIZES_STRING
    )
    ft_group.add_argument(
        "--min-mapq",
        "-mq",
        default=10,
        type=int,
        dest="min_mapq",
        help="Threshold for minimal mapping quality. Read with "
             "lower MAPQ will be discarded. Default: 10"
    )
    ft_group.add_argument(
        "--statistics",
        dest="statistics",
        action="store_true",
        default=False,
        help="Generate statistical values for window features."
    )

    parser.set_defaults(execute=run_feature_generation)

    return subparsers
# fmt: on


def filter_read_alignments(count_statistics, min_mapq, read):

    read_position = None
    if read.is_unmapped:
        count_statistics[ftms.UNMAPPED] += 1
    else:
        count_statistics[ftms.MAPPED] += 1
        if read.is_supplementary or read.is_secondary or read.is_qcfail:
            count_statistics[ftms.SUPPLEMENTARY] += 1
        elif read.is_duplicate:
            count_statistics[ftms.DUPLICATE] += 1
        elif read.mapping_quality < min_mapq:
            count_statistics[ftms.MAPQ] += 1
        elif read.is_read2:
            count_statistics[ftms.READ2] += 1
        else:
            count_statistics[ftms.GOOD] += 1
            read_position = read.reference_start
            if read.is_reverse:
                read_position *= -1
    return read_position


def finalize_mapping_statistics_per_chromosome(library_name, chromosome, mapping_stats):

    # add total read count for this chromosome
    mapping_stats[ftms.TOTAL] = mapping_stats[ftms.UNMAPPED] + mapping_stats[ftms.MAPPED]
    # turn into pd.Series
    mapping_stats = pd.Series(mapping_stats, dtype=np.int32)
    # cast names from enum to string for later storing as HDF5
    mapping_stats.index = [e.name for e in mapping_stats.index]
    mapping_stats_relative = mapping_stats / mapping_stats[ftms.TOTAL.name]
    mapping_stats = pd.DataFrame(
        [mapping_stats, mapping_stats_relative],
        index=pd.MultiIndex.from_tuples(
            [
                (library_name, chromosome, scales.ABSOLUTE.name),
                (library_name, chromosome, scales.RELATIVE.name),
            ],
            names=["library", "region", "scale"],
        ),
    )
    return mapping_stats


def collect_window_statistics_per_chromosome(
    crick_reads, watson_reads, window_sizes, chromosome, chrom_size, library_name
):

    collect_window_statistics = []
    window_statistics_index = []

    # generic bins [0...10...20.......100]
    # to count number of windows having
    # <10, <20, <30 ... percent of Watson reads
    decile_bins = np.arange(0, 101, 10)
    decile_labels = ftws.get_window_labels()
    assert len(decile_labels) == 10

    for w in window_sizes:
        # number of windows needs to be
        # recorded for data normalization
        num_windows = 0
        num_nonzero_windows = 0

        watson_decile_bins_abs = np.zeros(len(decile_labels), dtype=np.int32)
        for step in [0, w // 2]:
            # adjust chromosome size to include last incomplete
            # window; for larger window sizes such as 5 Mbp, this
            # can be a considerable fraction of the chromosome
            adjusted_chrom_size = (chrom_size + step) // w * w + w

            # +1 here to have last window created
            windows = np.arange(step, adjusted_chrom_size + 1, w)

            # check that start of last window is always smaller than
            # end of current chromosome
            assert (
                windows[-2] < chrom_size
            ), f"{chromosome} invalid window start: {windows[-2:]} >= {chrom_size} [{adjusted_chrom_size} / {w} / {step}]"
            # check that end of last window is always overlapping
            # end of current chromosome
            assert (
                windows[-1] >= chrom_size
            ), f"{chromosome} invalid window end: {windows[-2:]} < {chrom_size} [{adjusted_chrom_size} / {w} / {step}]"

            # windows.size is length of list, but we need
            # number of bins here, i.e. -1
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
        counts = pd.Series(extended_decile_bins, index=ftws.get_extended_labels(), dtype=np.int32)
        collect_window_statistics.append(counts)
        window_statistics_index.append((library_name, chromosome, w, scales.ABSOLUTE.name))
        relative = counts / counts[ftws.NZERO.name]
        relative[[ftws.NZERO.name, ftws.TOTAL.name]] = (
            counts[[ftws.NZERO.name, ftws.TOTAL.name]] / counts[ftws.TOTAL.name]
        )
        collect_window_statistics.append(relative)
        window_statistics_index.append((library_name, chromosome, w, scales.RELATIVE.name))

    collect_window_statistics = pd.concat(collect_window_statistics, axis=1, ignore_index=False)
    collect_window_statistics = collect_window_statistics.transpose()
    collect_window_statistics.index = pd.MultiIndex.from_tuples(
        window_statistics_index, names=["library", "region", "window_size", "scale"]
    )
    return collect_window_statistics


def normalize_library_name(library_name):

    if library_name.isidentifier():
        pass
    else:
        original_name = library_name
        library_name = library_name.replace("-", "_")
        library_name = library_name.replace(".", "_")
        library_name = library_name.replace(":", "_")
        library_name = library_name.replace("/", "_")
        library_name = library_name.replace(" ", "_")
        if not library_name.isidentifier():
            raise ValueError(
                "The library name cannot be normalized to a valid "
                f"Python identifier: {original_name} >>> {library_name}"
            )
    return library_name


def compute_features_per_chromosome(parameter_set):
    """
    Parameter feature types is unused at the moment
    """

    bam_file, chromosome, window_sizes, min_mapq, feature_types = parameter_set
    library_name = normalize_library_name(pl.Path(bam_file.stem).name)

    mapping_stats = col.Counter()
    aln_filter = fnt.partial(filter_read_alignments, mapping_stats, min_mapq)

    with pysam.AlignmentFile(bam_file, "r") as bam:
        chrom_size = bam.get_reference_length(chromosome)
        try:
            # create int array of alignment start positions for good reads only
            good_reads = np.array(
                [a for a in map(aln_filter, bam.fetch(chromosome)) if a is not None],
                dtype=np.int32,
            )
        except ValueError as err:
            raise ValueError(f"Cannot load reads from file {str(bam_file)}: {str(err)}")
        crick_reads = good_reads[good_reads > 0]  # "plus" reads
        watson_reads = good_reads[good_reads < 0] * -1  # "minus" reads

    mapping_stats = finalize_mapping_statistics_per_chromosome(
        library_name, chromosome, mapping_stats
    )

    window_stats = collect_window_statistics_per_chromosome(
        crick_reads, watson_reads, window_sizes, chromosome, chrom_size, library_name
    )

    results = {
        "library_name": library_name,
        "chromosome": chromosome,
        "mapping_statistics": mapping_stats,
        "window_statistics": window_stats,
        "crick_reads": pd.Series(crick_reads, dtype=np.int32),
        "watson_reads": pd.Series(watson_reads, dtype=np.int32),
    }

    return results


def aggregate_mapping_features(feature_set, num_chromosomes):
    """ """
    libraries = feature_set.index.unique(level="library")

    wg_aggregates = []

    for lib in libraries:
        counts = feature_set.xs([lib, scales.ABSOLUTE.name], level=["library", "scale"])
        assert counts.shape[0] == num_chromosomes
        counts = counts.sum(axis=0)
        norm_subset = counts / counts[ftms.TOTAL.name]
        df = pd.DataFrame(
            [counts, norm_subset],
            columns=counts.index,
            index=pd.MultiIndex.from_tuples(
                [
                    (lib, "wg", scales.ABSOLUTE.name),
                    (lib, "wg", scales.RELATIVE.name),
                ],
                names=["library", "region", "scale"],
            ),
        )
        wg_aggregates.append(df)
    wg_aggregates = pd.concat(wg_aggregates, axis=0, ignore_index=False)
    return wg_aggregates


def aggregate_window_features(feature_set, num_chromosomes):
    """ """
    libraries = feature_set.index.unique(level="library")
    window_sizes = feature_set.index.unique(level="window_size")

    wg_aggregates = []

    for lib in libraries:
        for window in window_sizes:
            counts = feature_set.xs(
                [lib, window, scales.ABSOLUTE.name], level=["library", "window_size", "scale"]
            )
            assert counts.shape[0] == num_chromosomes
            counts = counts.sum(axis=0)
            norm_subset = counts / counts[ftws.NZERO.name]
            norm_subset[[ftws.NZERO.name, ftws.TOTAL.name]] = (
                counts[[ftws.NZERO.name, ftws.TOTAL.name]] / counts[ftws.TOTAL.name]
            )
            df = pd.DataFrame(
                [counts, norm_subset],
                columns=counts.index,
                index=pd.MultiIndex.from_tuples(
                    [
                        (lib, "wg", window, scales.ABSOLUTE.name),
                        (lib, "wg", window, scales.RELATIVE.name),
                    ],
                    names=["library", "region", "window_size", "scale"],
                ),
            )
            wg_aggregates.append(df)
    wg_aggregates = pd.concat(wg_aggregates, axis=0, ignore_index=False)
    return wg_aggregates


def compute_features(
    input_bams,
    chromosomes,
    window_sizes,
    min_mapq,
    feature_types,
    discard_positions,
    jobs,
    out_hdf,
):
    """
    Parameter feature_types not used at the moment
    """

    collect_mapping_stats = []
    collect_window_stats = []

    with mp.Pool(jobs) as pool:
        resiter = pool.imap_unordered(
            compute_features_per_chromosome,
            create_parameter_combination(
                input_bams, chromosomes, window_sizes, min_mapq, feature_types
            ),
        )
        for results in resiter:
            if not discard_positions:
                key_crick_reads = f"reads/{results['library_name']}/{results['chromosome']}/crick"
                key_watson_reads = (
                    f"reads/{results['library_name']}/{results['chromosome']}/watson"
                )
                results["crick_reads"].to_hdf(
                    out_hdf, key_crick_reads, mode="a", complevel=9, complib="blosc"
                )
                results["watson_reads"].to_hdf(
                    out_hdf, key_watson_reads, mode="a", complevel=9, complib="blosc"
                )
            collect_mapping_stats.append(results["mapping_statistics"])
            collect_window_stats.append(results["window_statistics"])

    logger.debug("Concatenating per-library/per-chromosome features")
    collect_mapping_stats = pd.concat(collect_mapping_stats, axis=0, ignore_index=False)
    logger.debug(f"Mapping statistics result dimension: {collect_mapping_stats.shape}")
    collect_window_stats = pd.concat(collect_window_stats, axis=0, ignore_index=False)
    logger.debug(f"Window statistics result dimension: {collect_window_stats.shape}")

    logger.debug("Aggregating per-chromosome features...")
    wg_mapping_stats = aggregate_mapping_features(collect_mapping_stats, len(chromosomes))
    logger.debug("Whole-genome mapping statistics aggregated")

    wg_window_stats = aggregate_window_features(collect_window_stats, len(chromosomes))
    logger.debug("Whole-genome window statistics aggregated")

    collect_mapping_stats = pd.concat(
        [collect_mapping_stats, wg_mapping_stats], axis=0, ignore_index=False
    )
    key_mapping_stats = "features/mapping_statistics"
    collect_mapping_stats.sort_index(inplace=True)
    collect_mapping_stats.to_hdf(
        out_hdf, key_mapping_stats, mode="a", complevel=9, complib="blosc"
    )
    logger.debug("Stored mapping statistics features...")
    collect_window_stats = pd.concat(
        [collect_window_stats, wg_window_stats], axis=0, ignore_index=False
    )
    key_window_stats = "features/window_statistics"
    collect_window_stats.sort_index(inplace=True)
    collect_window_stats.to_hdf(out_hdf, key_window_stats, mode="a", complevel=9, complib="blosc")
    logger.debug("Stored window statistics features...")
    return collect_mapping_stats, collect_window_stats


def collect_input_bam_files(input_paths, recursive, bam_ext):
    """
    input_paths: list of strings; single or multiple files, folders or
    a mix of file and folder paths
    recursive: boolean; recurse into subfolders
    """
    assert isinstance(input_paths, list), f"Input_paths is not a list: {type(input_paths)}"

    bam_files = []
    bam_ext = bam_ext.strip('"')
    if not bam_ext.startswith("."):
        bam_ext = "." + bam_ext
    glob_ext = f"*{bam_ext}"

    for path in input_paths:
        if path.is_file():
            logger.debug(f"Adding BAM file {str(path)}")
            bam_files.append(path)
        elif path.is_dir():
            logger.debug(f"Checking folder... {str(path)}")
            if recursive:
                logger.debug("...recursively")
                bam_files.extend(path.rglob(glob_ext))
            else:
                logger.debug("...non-recursively")
                bam_files.extend(path.glob(glob_ext))
        else:
            raise ValueError(f"Cannot handle input path: {str(path)}")

    if not bam_files:
        raise ValueError(
            f"No input BAM files collected with pattern >{bam_ext}< "
            f"underneath path >{str(input_paths[0])}<"
        )

    missing_index_files = []
    for bam_file in bam_files:
        if not bam_file.with_suffix(".bam.bai").is_file():
            missing_index_files.append(str(bam_file))

    if missing_index_files:
        missing_index_files = "\n".join(sorted(missing_index_files)) + "\n"
        raise ValueError(f"Index (.bam.bai) files missing for files: {missing_index_files}")

    return sorted(bam_files)


def collect_chromosomes(bam_files, chrom_match_expr):
    """
    Extract info which chromosomes are present in BAM files
    (sample at most 5 at random)
    """
    chrom_match = re.compile(chrom_match_expr.strip('"'))
    logger.debug(f"Using the following expression to select chromosomes: {chrom_match.pattern}")

    select_chromosomes = set()

    attempts = min(len(bam_files), 5)

    for _ in range(attempts):
        bam_file = rand.choice(bam_files)
        logger.debug(f"Loading chromosome info from file {str(bam_file)}")
        with pysam.AlignmentFile(bam_file, "r") as bam:
            file_chromosomes = [
                chrom for chrom in bam.references if chrom_match.match(chrom) is not None
            ]
            if not select_chromosomes:
                select_chromosomes = select_chromosomes.union(set(file_chromosomes))
            else:
                if not all(c in select_chromosomes for c in file_chromosomes):
                    select_chromosomes = "\n".join(sorted(select_chromosomes)) + "\n"
                    file_chromosomes = "\n".join(sorted(file_chromosomes)) + "\n"
                    raise ValueError(
                        "Chromosome sets do not match: "
                        f"{select_chromosomes} ---\n {file_chromosomes}"
                    )

    if not select_chromosomes:
        raise ValueError("No chromosome information found in BAM file(s)")

    # shuffle once to avoid that all large chromosomes
    # are processed in parallel
    rand.shuffle(list(select_chromosomes))

    return select_chromosomes


def create_parameter_combination(input_bams, chromosomes, window_sizes, min_mapq, feature_types):
    """
    feature_types: types of features to compute - implemented for later use, not used at the moment
    """
    total_bam = len(input_bams)
    total_chroms = len(chromosomes)
    for n_bam, bam_file in enumerate(input_bams, start=1):

        for m_chrom, chromosome in enumerate(chromosomes, start=1):

            param_set = (bam_file, chromosome, window_sizes, min_mapq, feature_types)
            # logger.debug(
            #     f"Creating parameter set BAM {n_bam}/{total_bam} "
            #     f"and CHROM {m_chrom}/{total_chroms}"
            # )
            yield param_set
    return


def dump_output_table(mapping_stats, window_stats, out_path):

    wg_subset = mapping_stats.xs(["wg", scales.RELATIVE.name], level=["region", "scale"]).copy()
    wg_subset.drop(ftms.TOTAL.name, axis=1, inplace=True)
    feat_per_library = [wg_subset]

    for w in window_stats.index.unique(level="window_size"):
        wg_subset = window_stats.xs(
            ["wg", w, scales.RELATIVE.name], level=["region", "window_size", "scale"]
        ).copy()
        wg_subset.drop(ftws.TOTAL.name, axis=1, inplace=True)
        wg_subset.columns = [f"WS{w}_{c}" for c in wg_subset.columns]
        feat_per_library.append(wg_subset)

    feat_per_library = pd.concat(feat_per_library, axis=1, ignore_index=False)
    logger.debug(f"Dataset dimension dumped to TSV: {feat_per_library.shape}")
    feat_per_library.sort_index(inplace=True)

    feat_per_library.to_csv(out_path, sep="\t", header=True, index=True, index_label="library")
    return


def run_feature_generation(args):
    """ """
    if args.output_plotting:
        raise NotImplementedError("Generatig the plotting output file is not yet supported.")

    logger.info("Running feature generation module...")
    logger.info(f"Computing features for window size(s): {args.window_size}")
    mapq_threshold = args.min_mapq

    logger.debug("Checking output path...")
    args.output_features.parent.mkdir(exist_ok=True, parents=True)

    out_tsv = args.output_features
    out_hdf = args.output_features.with_suffix(".h5")
    logger.debug(f"Writing output feature table to path {out_tsv}")

    input_bams = collect_input_bam_files(args.input_bam, args.recursive, args.bam_extension)
    logger.info(f"Collected {len(input_bams)} BAM files to process")
    process_chroms = collect_chromosomes(input_bams, args.chromosomes)
    logger.info(f"Selected {len(process_chroms)} chromosomes to process")
    total_combinations = len(input_bams) * len(process_chroms)
    logger.info(f"Total number of parameter combinations to process: {total_combinations}")

    logger.info(f"Start feature computation using {args.jobs} CPU cores")
    mapping_stats, window_stats = compute_features(
        input_bams,
        process_chroms,
        args.window_size,
        mapq_threshold,
        tuple(),  # this is a placeholder for "feature_types"
        args.discard_read_positions,
        args.jobs,
        out_hdf,
    )
    logger.info("Feature computation completed, dumping TSV output table")

    _ = dump_output_table(mapping_stats, window_stats, out_tsv)

    logger.info("Done - exiting...")

    return
