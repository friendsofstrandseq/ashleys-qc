import glob
import fnmatch
from multiprocessing import Pool
from collections import Counter
from collections import defaultdict
import statistics
import os
import re
import logging

import pysam

logger = logging.getLogger(__name__)
CHROMOSOMES = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12',
               'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']

# fmt: off
def add_features_parser(subparsers):
    parser = subparsers.add_parser("features", help="Compute features for (indexed) Strand-seq BAM files")
    parser.add_argument(
        "--file",
        "-f",
        required=True,
        help="the name of the bam file to analyze or a directory where all bam files are processed",
    )
    parser.add_argument(
        "--window_size",
        "-w",
        type=int,
        nargs="+",
        required=True,
        help="window size for feature generation"
    )
    parser.add_argument(
        "--output_features",
        "-o",
        required=True,
        help="name of output file for feature table, should be .tsv"
    )
    parser.add_argument(
        "--output_plotting",
        "-p",
        default=None,
        help="name of output file for Watson percentage lists, further used for plotting"
    )
    parser.add_argument(
        "--bam_extension",
        "-e",
        help="Specify extension of files used for feature computation. Default: '.bam'",
        dest='bam_extension',
        default=".bam"
    )
    parser.add_argument(
        "--mapping_quality",
        "-mq",
        help="threshold for minimal mapping quality required. Default: 10",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--recursive_collect",
        dest="recursive",
        action="store_true",
        default=False,
        help="Assume. Default: only collecting bam files "
        "from current folder",
    )
    parser.add_argument(
        "--chromosomes",
        "-c",
        default="^(chr)?[0-9X]+$",
        help="regex expression specifying chromosomes to use for feature generation. Default: chromosomes 1-22, X",
    )
    parser.add_argument(
        "--statistics",
        dest="statistics",
        action="store_true",
        default=False,
        help="generate statistical values for window features, increases feature set",
    )

    parser.set_defaults(execute=run_feature_generation)

    return subparsers
# fmt: on


def get_chrom_header(window_list, match_chromosomes):
    chromosome_list = [x for x in CHROMOSOMES if match_chromosomes.match(x)]
    feature_names = []
    wc_features = [
        "W10",
        "W20",
        "W30",
        "W40",
        "W50",
        "W60",
        "W70",
        "W80",
        "W90",
        "W100",
    ]
    mb = 1000000
    for w in window_list:
        for c in chromosome_list:
            chrom_features = [c + '_' + wc + '_' + str(round(w / mb, 2)) + "mb" for wc in wc_features]
            feature_names = feature_names + chrom_features

    return feature_names


def get_header(windows_list, use_statistics):
    feature_names = []
    regular_features = [
        "W10",
        "W20",
        "W30",
        "W40",
        "W50",
        "W60",
        "W70",
        "W80",
        "W90",
        "W100",
        "total_nonempty_windows",
        "total_nonempty_empty_windows",
    ]
    if use_statistics:
        statistics_features = ["stdev", "mean", "n_stdev", "n_mean"]
        regular_features = regular_features + statistics_features
    mb = 1000000
    for w in windows_list:
        for f in regular_features:
            feature_names.append(f + "_" + str(round(w / mb, 2)) + "mb")

    constant_features = [
        "p_unmap",
        "p_map",
        "p_supp",
        "p_dup",
        "p_mq",
        "p_read2",
        "p_good",
        "sample_name",
    ]
    feature_names = feature_names + constant_features
    return feature_names


def get_statistics(f_list):
    feature_list = []
    if len(f_list) < 2:
        feature_list = feature_list + ["0", "0", "0", "0"]
        return feature_list

    feature_list.append(str(statistics.stdev(f_list)))
    feature_list.append(str(statistics.mean(f_list)))
    feature_list.append(str(statistics.variance(f_list)))
    feature_list.append(str(statistics.median(f_list)))

    return feature_list


def get_w_percentage(total_window_collection_wc, total_window_collection):
    """
    function needed to plot distribution of Watson percentage for all windows
    currently used by plotting.py with --w_percentage
    currently only saved in output file for the last (smallest) window size
    note: new output file should also contain sample names (like the output feature table)

    :param total_window_collection_wc: containing number of Watson and Crick reads for each window
    :param total_window_collection: containing total number of reads for each window
    :return: list containing Watson read percentage for each window
    """
    raise NotImplementedError


def get_wc_composition(total_window_collection_wc, total_window_collection, window_count, chromosome_list):
    """
    TODO: rewrite this (@PE)
    """
    # create 10 features for 10% steps of w-percentage in windows
    feature_list = []
    window_dict = dict(total_window_collection)
    wc_collection = Counter(W10=0, W20=0, W30=0, W40=0, W50=0, W60=0, W70=0, W80=0, W90=0, W100=0)
    values = []
    # total: counts non-empty windows
    total = 0
    cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    wc_difference = []  # this is no longer used outside of this function
    last_window = -1
    current_window = -1
    # w_percentage_list is a data structure only used for
    # plotting (see plotting module, "--w_percentage" parameter)
    w_percentage_list = []
    chromosome_wc = Counter()

    for i in window_dict.keys():
        # calculate wc composition of whole sample dependent on percentage of w strands in windows
        if total_window_collection_wc[i] <= 1:
            continue
        # not entirely clear: -1 ?
        w_percentage = (total_window_collection_wc[i + "W"] - 1) / (
            total_window_collection_wc[i] - 1
        )
        current_chrom = i.split('_')[0]
        w_percentage_list.append(str(w_percentage))
        for j in cuts:
            if j <= w_percentage < j + 0.1:
                c = "W" + str(int((j + 0.1) * 100))
                wc_collection[c] += 1
                chrom = current_chrom + c
                chromosome_wc[chrom] += 1
                total += 1
                current_window = j
        if w_percentage >= 1:
            c = "W" + str(int(100))
            wc_collection[c] += 1
            chrom = current_chrom + c
            chromosome_wc[chrom] += 1
            total += 1
        # compute difference in WC-ratio binning
        # for subsequent windows
        # Currently discarded after this function
        # NB: not the same as the difference in total
        # read count computed in "get_read_features"
        if not last_window == -1:
            wc_difference.append(last_window - current_window)
        last_window = current_window

        values.append(total_window_collection[i])

    # next if-else:
    # normalize counts in wc_collection
    # (share of windows with 10% W, 20 % W etc.)
    # by total number of non-empty windows
    chromosome_feature_list = []
    if total == 0:
        zero_list = ["0.0"] * 10
        feature_list = feature_list + zero_list
        zero_list = ['0.0'] * 10 * len(chromosome_list) #23 chromosomes x 10 features
        chromosome_feature_list = chromosome_feature_list + zero_list

    else:
        for i in range(0, 100, 10):
            c = "W" + str(i + 10)
            feature_list.append(str(wc_collection[c]))
        for chrom in chromosome_list:
            for i in range(0, 100, 10):
                c = chrom + "W" + str(i + 10)
                chromosome_feature_list.append(str(chromosome_wc[c]))


    feature_list.append(str(total))
    feature_list.append(str(window_count))

    return values, wc_difference, w_percentage_list, feature_list, chromosome_feature_list


def get_read_features(chrom, bamfile_name, window_size, mapq_threshold):
    """
    TODO: rewrite this (@PE)
    """
    count_collection = Counter()
    with pysam.AlignmentFile(bamfile_name, "rb") as bamfile:
        chromosomes = dict(zip(bamfile.references, bamfile.lengths))
        length = chromosomes.get(chrom)
        step_size = int(window_size / 2)

        window_collection = Counter()
        window_collection_wc = Counter()
        neighbor_difference = []
        window_count = 0

        # count reads in each window of size stepsize
        for i in range(0, length, step_size):
            window_count += 1
            s = str(chrom) + '_' + str(i)
            window_collection[s] += 1
            window_collection_wc[s] += 1
            window_collection_wc[s + "W"] += 1
            window_collection_wc[s + "C"] += 1
            for read in bamfile.fetch(chrom, i, i + window_size):
                window_collection[s] += 1

                # NB: all of the following if's count constant feature
                # values per chromosome, i.e. counting reads twice
                # is avoided by shifting: i+step_size
                if read.is_unmapped:
                    if read.reference_start > i + step_size or i == 0:
                        count_collection["unmapped"] += 1
                    continue
                # count all mapped reads
                if read.reference_start > i + step_size or i == 0:
                    count_collection["mapped"] += 1
                if read.is_supplementary or read.is_secondary or read.is_qcfail:
                    if read.reference_start > i + step_size or i == 0:
                        count_collection["supplementary"] += 1
                        # TODO: most likely not used anymore
                        window_collection_wc[s + "_supp"] += 1
                    continue
                if read.is_duplicate:
                    if read.reference_start > i + step_size or i == 0:
                        count_collection["duplicate"] += 1
                    continue
                if read.mapping_quality < mapq_threshold:
                    if read.reference_start > i + step_size or i == 0:
                        count_collection["mapping_quality"] += 1
                    continue
                if read.is_read2:
                    if read.reference_start > i + step_size or i == 0:
                        count_collection["read2"] += 1
                    continue
                if read.reference_start > i + step_size or i == 0:
                    count_collection["good"] += 1

                # NB: i is not shifted by step_size, i.e. we count
                # reads that overlap two windows twice
                window_collection_wc[s] += 1
                if read.is_reverse:
                    window_collection_wc[s + "W"] += 1
                else:
                    window_collection_wc[s + "C"] += 1

            if not i == 0:
                last_window = str(chrom) + str(i - step_size)
                # difference in total number of reads (any quality) between
                # subsequent windows
                diff = window_collection[last_window] - window_collection[s]
                neighbor_difference.append(diff)

    return (
        chrom,
        count_collection,
        window_collection,
        window_collection_wc,
        neighbor_difference,
        window_count,
    )


def get_bam_characteristics(
    jobs, window_list, bamfile_name, mapq_threshold, match_chromosomes, use_statistics
):
    """
    TODO: rewrite this (@PE)
    """
    # read a BAM file and return different features for windows of the chromosomes

    with pysam.AlignmentFile(bamfile_name, "rb") as bamfile:
        references = bamfile.references
        chromosome_list = [x for x in references if match_chromosomes.match(x)]
        logger.debug(
            "Chromosomes used for feature generation: {}".format(
                ", ".join(sorted(chromosome_list))
            )
        )

    filtered_list = [
        "unmapped",
        "mapped",
        "supplementary",
        "duplicate",
        "mapping_quality",
        "read2",
        "good",
    ]
    feature_list = []
    chromosome_feature_list = []

    # collections with different counts over all chromosomes
    for w_size in window_list:
        window_size = w_size
        total_count_collection = Counter([])
        total_window_collection = Counter([])
        total_window_collection_wc = Counter([])
        total_neighbor_difference = []

        p = Pool(jobs)
        args_list = [(c, bamfile_name, window_size, mapq_threshold) for c in chromosome_list]
        result = p.starmap(get_read_features, args_list)
        p.close()
        p.join()
        window_count = 0
        for r in result:
            total_count_collection += r[1]
            total_window_collection += r[2]
            total_window_collection_wc += r[3]
            total_neighbor_difference += r[4]
            window_count += r[5]

        # values: total read count per window, sorted by window id
        values, wc_difference, w_percentage_list, next_features, next_chromosomes= get_wc_composition(
            total_window_collection_wc, total_window_collection, window_count, chromosome_list
        )
        feature_list = feature_list + next_features
        chromosome_feature_list = chromosome_feature_list + next_chromosomes
        if use_statistics:
            statistics_features = get_statistics(values)
            neighbor_features = get_statistics(total_neighbor_difference)
            feature_list = feature_list + statistics_features + neighbor_features

    # relative filtering feature values
    total_reads = total_count_collection["mapped"] + total_count_collection["unmapped"]
    for i in filtered_list:
        feature_list.append(str(total_count_collection[i] / total_reads))

    # add sample name
    file_name = os.path.basename(bamfile_name)
    feature_list.append(file_name)

    return w_percentage_list, feature_list, chromosome_list, chromosome_feature_list


def collect_features(
    jobs,
    windowsize_list,
    bamfile,
    mapq_threshold,
    output,
    distribution_file,
    match_chromosomes,
    use_statistics,
    chrom_output
):
    w_list, feature_list, chromosomes, chromosome_features = get_bam_characteristics(
        jobs, windowsize_list, bamfile, mapq_threshold, match_chromosomes, use_statistics
    )
    # distribution file: only used for plotting
    distribution_file.write("\t".join(w_list))
    distribution_file.write("\n")
    # main output file: feature values per library
    output.write("\t".join(feature_list))
    output.write("\n")
    # output file with features for each chromosome
    chrom_output.write("\t".join(chromosome_features))
    chrom_output.write("\n")


def collect_input_bam_files(input_path, recursive_collect, file_ext):
    """
    input_path: single file path or folder
    """
    input_files = []
    if os.path.isfile(input_path):
        logger.debug("Single input file detected: {}".format(input_path))
        input_files = [input_path]
    elif os.path.isdir(input_path):
        if recursive_collect:
            logger.debug("Collect input files via recursive collect")
            match_pattern = "*" + file_ext
            for root, _, files in os.walk(input_path, followlinks=False):
                matched_files = fnmatch.filter(files, match_pattern)
                input_files.extend([os.path.join(root, f) for f in matched_files])
        else:
            logger.debug("Collect input files via glob expand")
            match_pattern = os.path.join(input_path, "*" + file_ext)
            input_files = glob.glob(match_pattern)
    else:
        raise ValueError("Path is neither single input file nor folder: {}".format(input_path))

    input_files = sorted([os.path.abspath(f) for f in input_files])

    if not input_files:
        raise ValueError(
            "No input files selected with parameters: {} / [recursive:] {} / {}".format(
                input_path, recursive_collect, file_ext
            )
        )

    return input_files


def run_feature_generation(args):
    """ """
    if args.output_plotting:
        raise NotImplementedError

    logger.info("Running feature generation module...")
    logger.info("calculating absolute values for Watson features, additional value for number of empty and non-empty windows")
    windowsize_list = args.window_size
    windowsize_list.sort(reverse=True)
    mapq_threshold = args.mapping_quality

    output_file = args.output_features

    file_name, ending = output_file.rsplit(".", 1)
    distribution_file = open(file_name + "_window_distribution." + ending, "w")
    output = open(output_file, "w")
    features = get_header(windowsize_list, args.statistics)
    output.write("\t".join(features))
    output.write("\n")

    match_chrom_expression = args.chromosomes.strip('"')

    logger.debug(
        "Using following expression to match chromosomes: {}".format(match_chrom_expression)
    )
    match_chromosomes = re.compile(match_chrom_expression)

    chrom_output = open(file_name + '_chromosomes.tsv', "w")
    chrom_features = get_chrom_header(windowsize_list, match_chromosomes)
    chrom_output.write("\t".join(chrom_features))
    chrom_output.write("\n")

    logger.debug("Collecting input files...")
    input_files = collect_input_bam_files(args.file, args.recursive, args.bam_extension.strip('"'))
    num_input_files = len(input_files)
    logger.info("Identified {} input BAM files".format(num_input_files))
    logger.debug("Head of input file list:\n{}".format("\n".join(input_files[:3])))

    for pos, input_file in enumerate(input_files, start=1):
        logger.info(
            "Processing input file ({}/{}): {}".format(
                pos, num_input_files, os.path.basename(input_file)
            )
        )
        collect_features(
            args.jobs,
            windowsize_list,
            input_file,
            mapq_threshold,
            output,
            distribution_file,
            match_chromosomes,
            args.statistics,
            chrom_output
        )

    output.close()
    distribution_file.close()
    chrom_output.close()
    logger.info("Output files closed")
    return
