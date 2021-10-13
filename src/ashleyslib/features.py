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

from ashleyslib import FeatureTypes as FTYPES
from ashleyslib import Columns, Units, GenomicRegion, Orientation

import ashleyslib.ft_mapstats as ftms
import ashleyslib.ft_wincount as ftwcnt
import ashleyslib.ft_windiv as ftwdiv


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
        "--feature-types",
        "-ft",
        dest="feature_types",
        nargs='+',
        type=str,
        default=['window_counts', 'window_divergence'],
        help="Specify feature types to compute (besides basic mapping statistics). "
             "Default: window_counts window_divergence"
    )

    parser.set_defaults(execute=run_feature_generation)

    return subparsers
# fmt: on


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
    """ """

    bam_file, chromosome, window_sizes, min_mapq, compute_features = parameter_set
    library_name = normalize_library_name(pl.Path(bam_file.stem).name)

    mapping_stats = col.Counter()
    aln_filter = fnt.partial(ftms.filter_read_alignments, mapping_stats, min_mapq)

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

    # note that the basic mapping statistics features are not optional
    # b/c they are essentially computed entirely while reading the read alignments
    mapping_stats = ftms.finalize_mapstats_per_chromosome(library_name, chromosome, mapping_stats)

    results = {
        Columns.library: library_name,
        Columns.region: chromosome,
        FTYPES.mapstats: mapping_stats,
    }

    # TODO: if all feature compute funtions can be implemented with the same
    # interface, this could be abstracted here...
    if FTYPES.window_counts in compute_features:
        window_count_stats = ftwcnt.collect_window_count_statistics_per_chromosome(
            crick_reads, watson_reads, window_sizes, chromosome, chrom_size, library_name
        )
        results[FTYPES.wincounts] = window_count_stats

    if FTYPES.window_divergence in compute_features:
        window_divergence_stats = ftwdiv.collect_window_divergence_statistics_per_chromosome(
            crick_reads, watson_reads, window_sizes, chromosome, chrom_size, library_name
        )
        results[FTYPES.windiv] = window_divergence_stats

    results[Orientation.crick] = pd.Series(crick_reads, dtype=np.int32)
    results[Orientation.watson] = pd.Series(watson_reads, dtype=np.int32)

    return results


def aggregate_features(parameter_set):

    ftype, records, num_chromosomes = parameter_set

    records = pd.concat(records, axis=0, ignore_index=False)
    if ftype == FTYPES.mapstats:
        wg_records, packed_table = ftms.aggregate_mapstats_genome_wide(records, num_chromosomes)
        records = pd.concat([records, wg_records], axis=0, ignore_index=False)
    elif ftype == FTYPES.window_count:
        wg_records, packed_table = ftwcnt.aggregate_window_count_features(records, num_chromosomes)
        records = pd.concat([records, wg_records], axis=0, ignore_index=False)
    elif ftype == FTYPES.window_divergence:
        records, packed_table = ftwdiv.aggregate_window_divergence_features(
            records, num_chromosomes
        )
    else:
        raise ValueError(f"Cannot process feature type {ftype}")
    # important: the records DF may contain Enum-type objects that cannot be loaded
    # outside of ASHLEYS. These must be cast to string before dumping to HDF
    index_columns = [str(level.name) for level in records.index.levels]
    records.reset_index(inplace=True, drop=False)
    records.columns = [str(c) for c in records.columns]
    records[index_columns] = records[index_columns].astype(str)
    records.set_index(index_columns, inplace=True, drop=True)
    records.sort_index(inplace=True)
    return ftype, records, packed_table


def compute_features(
    input_bams,
    chromosomes,
    window_sizes,
    min_mapq,
    compute_features,
    discard_positions,
    jobs,
    out_hdf,
):
    """ """
    collect_feature_stats = col.defaultdict(list)

    with mp.Pool(jobs) as pool:
        resiter = pool.imap_unordered(
            compute_features_per_chromosome,
            create_parameter_combination(
                input_bams, chromosomes, window_sizes, min_mapq, compute_features
            ),
        )
        for results in resiter:
            if not discard_positions:
                key_crick_reads = f"reads/{results[Columns.library]}/{results[Columns.region]}/{Orientation.crick}"
                key_watson_reads = f"reads/{results[Columns.library]}/{results[Columns.region]}/{Orientation.watson}"

                results[Orientation.crick].to_hdf(
                    out_hdf, key_crick_reads, mode="a", complevel=9, complib="blosc"
                )
                results[Orientation.watson].to_hdf(
                    out_hdf, key_watson_reads, mode="a", complevel=9, complib="blosc"
                )
            collect_feature_stats[FTYPES.mapping_stats].append(results[FTYPES.mapping_stats])
            for ftype in compute_features:
                collect_feature_stats[ftype].append(results[ftype])

    logger.debug("Feature computation per chromosome completed, starting aggregation...")

    packed_tables = []
    with mp.Pool(min(jobs, len(collect_feature_stats))) as pool:
        resiter = pool.imap_unordered(
            aggregate_features,
            [
                (ftype, records, len(chromosomes))
                for ftype, records in collect_feature_stats.items()
            ],
        )
        for ftype, agg_records, packed_table in resiter:
            logger.debug(f"Aggregation for feature {ftype} completed")
            key_features = f"features/{ftype}"
            agg_records.to_hdf(out_hdf, key_features, mode="a", complevel=9, complib="blosc")
            logger.debug(f"Stored features of type {ftype}")
            packed_tables.append(packed_table)
            logger.debug(f"Caching packed table of dimension {packed_table.shape}")

    logger.debug("Concatenating packed tables for text output")
    packed_tables = pd.concat(packed_tables, axis=1, ignore_index=False)
    packed_tables.sort_index(inplace=True)
    logger.debug(f"Size (ROW x COL) of final feature table: {packed_tables.shape}")
    return packed_tables


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
    """ """
    total_bam = len(input_bams)
    total_combinations = total_bam * len(chromosomes)
    combination = 0
    notify_thresholds = np.arange(0.1, 1, 0.1, dtype=np.float16)
    notify_thresholds = np.append(notify_thresholds, 0.95)
    notified = np.zeros(notify_thresholds.size, dtype=np.bool)
    for n_bam, bam_file in enumerate(input_bams, start=1):

        for _, chromosome in enumerate(chromosomes, start=1):

            param_set = (bam_file, chromosome, window_sizes, min_mapq, feature_types)
            combination += 1
            progress = combination / total_combinations
            select = np.isclose(progress, notify_thresholds, atol=0.005)
            if select.any():
                notify_idx = np.flatnonzero(select)[0]
                if not notified[notify_idx]:
                    notified[notify_idx] = True
                    logger.debug(
                        f"Processed ~{int(round(progress * 100, 0))}% of jobs "
                        f"(BAM file {n_bam} of total {total_bam})"
                    )
            yield param_set
    return


def validate_feature_types(feature_types):

    invalid_feature_types = []
    valid_feature_types = []
    for ft in feature_types:
        try:
            enum_ft = FTYPES[ft.lower()]
            valid_feature_types.append(enum_ft)
        except KeyError:
            invalid_feature_types.append(ft)
    if invalid_feature_types:
        raise ValueError(
            f"The following feature types are unknown: {[str(i) for i in invalid_feature_types]}"
        )
    return valid_feature_types


def run_feature_generation(args):
    """ """
    if args.output_plotting:
        raise NotImplementedError("Generatig the plotting output file is not yet supported.")

    logger.info("Running feature generation module...")
    logger.info(f"Computing features for window size(s): {args.window_size}")
    mapq_threshold = args.min_mapq

    logger.info("Checking feature types to compute...")
    feature_types = validate_feature_types(args.feature_types)
    logger.debug(f"Computing the following feature types: {[str(f) for f in feature_types]}")

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
    feature_table = compute_features(
        input_bams,
        process_chroms,
        args.window_size,
        mapq_threshold,
        feature_types,
        args.discard_read_positions,
        args.jobs,
        out_hdf,
    )
    logger.info("Feature computation completed, dumping TSV output table")

    feature_table.to_csv(out_tsv, sep="\t", header=True, index=True)

    logger.info("Done - exiting...")

    return
