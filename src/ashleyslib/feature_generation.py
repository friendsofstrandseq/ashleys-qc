import pysam
from multiprocessing import Pool
from collections import Counter
import statistics
import os
import re
import logging


def add_features_parser(subparsers):
    parser = subparsers.add_parser('features', help='create features for bam files')
    parser.add_argument('--file', '-f', required=True,
                        help='the name of the bam file to analyze or a directory where all bam files are processed')
    parser.add_argument('--window_size', '-w', help='window size for feature generation', type=int, required=True,
                        nargs='+')
    parser.add_argument('--output_file', '-o', help='name of output file, should be .tsv', required=True)
    parser.add_argument('--bam_extension', '-e', help='specify extension of files used for, default .bam',
                        default='.bam')
    parser.add_argument('--mapping_quality', '-mq', help='threshold for minimal mapping quality required, default:10',
                        default=10, type=int)
    parser.add_argument('--recursive_collect', dest='recursive', action='store_true', default=False,
                        help='collecting bam files from entire folder hierarchy, default: only collecting bam files '
                             'from current folder')
    parser.add_argument('--chromosomes', '-c', help='regex expression specifying chromosomes to use for feature '
                                                    'generation, default: chromosomes 1-22, X')
    parser.add_argument('--statistics', dest='statistics', action='store_true', default=False,
                        help='generate statistical values for window features, increases feature set')

    parser.set_defaults(execute=run_feature_generation)

    return subparsers


def get_header(windows_list, use_statistics):
    feature_names = []
    regular_features = ['W10', 'W20', 'W30', 'W40', 'W50', 'W60', 'W70', 'W80', 'W90', 'W100', 'total']
    if use_statistics:
        statistics_features = ['stdev', 'mean', 'var', 'median', 'n_stdev', 'n_mean', 'n_var', 'n_median']
        regular_features = regular_features + statistics_features
    mb = 1000000
    for w in windows_list:
        for f in regular_features:
            feature_names.append(f + '_' + str(round(w/mb, 2)) + 'mb')

    constant_features = ['p_unmap', 'p_map', 'p_supp', 'p_dup', 'p_mq', 'p_read2', 'p_good', 'sample_name']
    feature_names = feature_names + constant_features
    return feature_names


def get_statistics(f_list):
    feature_list = []
    if len(f_list) < 2:
        feature_list = feature_list + ['0', '0', '0', '0']
        return feature_list

    feature_list.append(str(statistics.stdev(f_list)))
    feature_list.append(str(statistics.mean(f_list)))
    feature_list.append(str(statistics.variance(f_list)))
    feature_list.append(str(statistics.median(f_list)))

    return feature_list


def get_wc_composition(total_window_collection_wc, total_window_collection, window_count):
    # create 10 features for 10% steps of w-percentage in windows
    feature_list = []
    window_dict = dict(total_window_collection)
    wc_collection = Counter(W10=0, W20=0, W30=0, W40=0, W50=0, W60=0, W70=0, W80=0, W90=0, W100=0)
    values = []
    total = 0
    cuts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    wc_difference = []
    last_window = -1
    current_window = -1
    w_percentage_list = []

    for i in sorted(window_dict.keys()):
        # calculate wc composition of whole sample dependent on percentage of w strands in windows
        if total_window_collection_wc[i] <= 1:
            continue
        w_percentage = (total_window_collection_wc[i+'W'] - 1)/(total_window_collection_wc[i] - 1)
        w_percentage_list.append(str(w_percentage))
        for j in cuts:
            if j <= w_percentage < j+0.1:
                c = 'W' + str(int((j+0.1)*100))
                wc_collection.update({c: 1})
                total += 1
                current_window = j
        if w_percentage >= 1:
            c = 'W' + str(int(100))
            wc_collection.update({c: 1})
            total += 1
        if not last_window == -1:
            wc_difference.append(last_window - current_window)
        last_window = current_window

        values.append(total_window_collection[i] - 1)

    if total == 0:
        zero_list = ['0.0'] * 10
        feature_list = feature_list + zero_list

    else:
        for i in range(0, 100, 10):
            c = 'W' + str(i+10)
            feature_list.append(str(wc_collection[c] / total))

    feature_list.append(str(total/window_count))

    return values, wc_difference, w_percentage_list, feature_list


def get_read_features(chrom, bamfile_name, window_size, mapq_threshold):

    count_collection = Counter([])
    with pysam.AlignmentFile(bamfile_name, "rb") as bamfile:
        chromosomes = dict(zip(bamfile.references, bamfile.lengths))
        length = chromosomes.get(chrom)
        step_size = int(window_size / 2)

        window_collection = Counter([])
        window_collection_wc = Counter([])
        neighbor_difference = []
        window_count = 0

        # count reads in each window of size stepsize
        for i in range(0, length, step_size):
            window_count += 1
            s = str(chrom) + ":" + str(i)
            window_collection.update({s: 1})
            window_collection_wc.update({s: 1})
            window_collection_wc.update({s+'W': 1})
            window_collection_wc.update({s+'C': 1})
            for read in bamfile.fetch(chrom, i, i+window_size):
                window_collection.update({s: 1})
                if read.is_unmapped:
                    if read.reference_start > i+step_size or i == 0:
                        count_collection.update({'unmapped': 1})
                    continue
                # count all mapped reads
                if read.reference_start > i + step_size or i == 0:
                    count_collection.update({'mapped': 1})
                if read.is_supplementary or read.is_secondary or read.is_qcfail:
                    if read.reference_start > i + step_size or i == 0:
                        count_collection.update({'supplementary': 1})
                        window_collection_wc.update({s+'_supp': 1})
                    continue
                if read.is_duplicate:
                    if read.reference_start > i + step_size or i == 0:
                        count_collection.update({'duplicate': 1})
                    continue
                if read.mapping_quality < mapq_threshold:
                    if read.reference_start > i + step_size or i == 0:
                        count_collection.update({'mapping_quality': 1})
                    continue
                if read.is_read2:
                    if read.reference_start > i + step_size or i == 0:
                        count_collection.update({'read2': 1})
                    continue
                if read.reference_start > i + step_size or i == 0:
                    count_collection.update({'good': 1})

                window_collection_wc.update({s: 1})
                if read.is_reverse:
                    window_collection_wc.update({s+'W': 1})
                else:
                    window_collection_wc.update({s+'C': 1})

            if not i == 0:
                last_window = str(chrom) + ":" + str(i-step_size)
                diff = window_collection[last_window] - window_collection[s]
                neighbor_difference.append(diff)

    return chrom, count_collection, window_collection, window_collection_wc, neighbor_difference, window_count


def get_bam_characteristics(jobs, window_list, bamfile_name, mapq_threshold, chromosomes, logging, use_statistics):
    # read a BAM file and return different features for windows of the chromosomes

    with pysam.AlignmentFile(bamfile_name, "rb") as bamfile:
        references = bamfile.references
        chrom_re = re.compile(chromosomes)
        chromosome_list = [x for x in references if chrom_re.match(x)]
        logging.info('Chromosomes used for feature generation: ' + str(chromosome_list))

    filtered_list = ['unmapped', 'mapped', 'supplementary', 'duplicate', 'mapping_quality', 'read2', 'good']
    feature_list = []

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

        values, wc_difference, w_percentage_list, next_features = get_wc_composition(total_window_collection_wc,
                                                                                     total_window_collection,
                                                                                     window_count)
        feature_list = feature_list + next_features
        if use_statistics:
            statistics_features = get_statistics(values)
            neighbor_features = get_statistics(total_neighbor_difference)
            feature_list = feature_list + statistics_features + neighbor_features

    # relative filtering feature values
    total_reads = total_count_collection['mapped'] + total_count_collection['unmapped']
    for i in filtered_list:
        feature_list.append(str(total_count_collection[i]/total_reads))

    # add sample name
    file_name = bamfile_name.rsplit('/', 1)[1]
    f_list = file_name.rsplit('_')
    if len(f_list) == 4:
        # for filenames like HG00268_hgsvc_ilnxs-80pe_01PE20433, the middle is omitted to get HG00268x01PE20433
        f1, f2, f3, f4 = f_list
        if f1.endswith('A') or f1.endswith('B'):
            f1 = f1[:-1]
        cell_name = f4.split('.', 1)[0]
        if cell_name.startswith('A') or cell_name.startswith('B'):
            cell_name = cell_name[1:]
        if cell_name.startswith('x'):
            feature_list.append(f1 + cell_name)
        else:
            feature_list.append(f1 + 'x' + cell_name)
    else:
        feature_list.append(file_name)

    return w_percentage_list, feature_list


def collect_features(jobs, windowsize_list, bamfile, mapq_threshold, output, distribution_file, chromosomes, log,
                     use_statistics):
    w_list, feature_list = get_bam_characteristics(jobs, windowsize_list, bamfile, mapq_threshold, chromosomes, log,
                                                   use_statistics)
    distribution_file.write("\t".join(w_list))
    distribution_file.write('\n')
    output.write("\t".join(feature_list))
    output.write('\n')


def run_feature_generation(args):
    windowsize_list = args.window_size
    windowsize_list.sort(reverse=True)
    mapq_threshold = args.mapping_quality

    path = args.file
    output_file = args.output_file

    file_name, ending = output_file.rsplit('.', 1)
    log_file = file_name + '.log'
    if args.logging is not None:
        log_file = args.logging
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO, handlers=[logging.FileHandler(log_file)])
    logging.info('Generating features for window sizes ' + str(windowsize_list))

    distribution_file = open(file_name + '_window_distribution.' + ending, 'w')
    output = open(output_file, 'w')
    features = get_header(windowsize_list, args.statistics)
    logging.info('list of all features to generate: ' + str(features))
    output.write('\t'.join(features))
    output.write('\n')

    chromosomes = "^(chr)?[0-9X]+$"  # chr1-22, chrX
    if args.chromosomes is not None:
        chromosomes = args.chromosomes

    if os.path.isfile(path):
        collect_features(args.jobs, windowsize_list, path, mapq_threshold, output, distribution_file, chromosomes,
                         logging, args.statistics)

    else:
        extension = args.bam_extension
        if not args.recursive:
            for name in os.listdir(path):
                if not name.endswith(extension):
                    continue
                next_cell = os.path.join(path, name)
                collect_features(args.jobs, windowsize_list, next_cell, mapq_threshold, output, distribution_file,
                                 chromosomes, logging, args.statistics)

        else:
            for root, subdirs, files in os.walk(path):
                for name in files:
                    if not name.endswith(extension):
                        continue
                    next_cell = os.path.join(root, name)
                    collect_features(args.jobs, windowsize_list, next_cell, mapq_threshold, output, distribution_file,
                                     chromosomes, logging, args.statistics)

    output.close()
    distribution_file.close()
    return
