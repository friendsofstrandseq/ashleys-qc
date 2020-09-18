

import pysam
from multiprocessing import Pool
from collections import Counter
import statistics
import os


def add_features_parser(subparsers):
    parser = subparsers.add_parser('features', help='create features for bam files')
    parser.add_argument('--jobs', '-j', help="the number of jobs used to generate features", type=int)
    parser.add_argument('--file', '-f', required=True,
                        help='the name of the bam file to analyze or a directory where all bam files are processed')
    parser.add_argument('--window_size', '-w', help='window size for feature generation', type=int, required=True,
                        nargs='+')
    parser.add_argument('--output_plot', '-p', help='create plot showing wc feature distribution', required=False)
    parser.add_argument('--output_file', '-o', help='name of output file, should be .tsv', required=True)
    parser.add_argument('--bam_extension', '-e', help='specify extension of files used for, default .bam')

    parser.set_defaults(execute=run_feature_generation)

    return subparsers


def get_header(windows_list):
    feature_names = []
    regular_features = ['W10', 'W20', 'W30', 'W40', 'W50', 'W60', 'W70', 'W80', 'W90', 'W100', 'total', 'stdev', 'mean',
                        'n_stdev', 'n_mean']
    mb = 1000000
    for w in windows_list:
        for f in regular_features:
            feature_names.append(f + '_' + str(round(w/mb, 2)) + 'mb')

    constant_features = ['unmap', 'map', 'supp', 'dup', 'mq', 'read2', 'good', 'p_unmap', 'p_map', 'p_supp', 'p_dup',
                         'p_mq', 'p_read2', 'p_good', 'sample_name']
    feature_names = feature_names + constant_features
    return feature_names


def get_statistics(list, all):
    feature_list = []
    if len(list) < 2:
        feature_list = feature_list + ['0', '0']
        if all:
            feature_list = feature_list + ['0', '0']
        return feature_list

    feature_list.append(str(statistics.stdev(list)))
    feature_list.append(str(statistics.mean(list)))
    if all:
        feature_list.append(str(statistics.variance(list)))
        feature_list.append(str(statistics.median(list)))

    return feature_list


def get_wc_composition(total_window_collection_wc, total_window_collection):
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
        if total_window_collection_wc[i] > 1:
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

        values.append(total_window_collection[i])

    if total == 0:
        zero_list = ['0.0'] * 10
        feature_list = feature_list + zero_list

    else:
        for i in range(0, 100, 10):
            c = 'W' + str(i+10)
            feature_list.append(str(wc_collection[c] / total))

    feature_list.append(str(total))

    return values, wc_difference, w_percentage_list, feature_list


def get_read_features(chrom, bamfile_name, window_size):
    mapq_threshold = 10

    count_collection = Counter([])
    with pysam.AlignmentFile(bamfile_name, "rb") as bamfile:
        chromosomes = dict(zip(bamfile.references, bamfile.lengths))
        length = chromosomes.get(chrom)
        step_size = int(window_size / 2)

        window_collection = Counter([])
        window_collection_wc = Counter([])
        neighbor_difference = []

        # count reads in each window of size stepsize
        for i in range(0, length, step_size):
            s = str(chrom) + str(i)
            window_collection.update({s:1})
            window_collection_wc.update({s: 1})
            window_collection_wc.update({s+'W': 1})
            window_collection_wc.update({s+'C': 1})
            for read in bamfile.fetch(chrom, i, i+window_size):
                window_collection.update({s:1})
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
                last_window = str(chrom) + str(i-step_size)
                diff = window_collection[last_window] - window_collection[s]
                neighbor_difference.append(diff)

    return chrom, count_collection, window_collection, window_collection_wc, neighbor_difference


def get_bam_characteristics(jobs, window_list, bamfile_name):
    # read a BAM file and return different features for windows of the chromosomes
    chromosome_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11',
                       'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21',
                       'chr22', 'chrX']

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
        args_list = [(c, bamfile_name, window_size) for c in chromosome_list]
        result = p.starmap(get_read_features, args_list)

        for r in result:
            # print(r)
            total_count_collection += r[1]
            total_window_collection += r[2]
            total_window_collection_wc += r[3]
            total_neighbor_difference += r[4]

        values, wc_difference, w_percentage_list, next_features = get_wc_composition(total_window_collection_wc,
                                                                                     total_window_collection)

        # print_statistics(wc_difference, True)
        statistics_features = get_statistics(values, False)
        neighbor_features = get_statistics(total_neighbor_difference, False)
        feature_list = feature_list + next_features + statistics_features + neighbor_features

    # absolute filtering feature values
    for i in filtered_list:
        feature_list.append(str(total_count_collection[i]))

    # relative filtering feature values
    total_reads = total_count_collection['mapped'] + total_count_collection['unmapped']
    for i in filtered_list:
        feature_list.append(str(total_count_collection[i]/total_reads))

    # add filename as sample+cell
    file = bamfile_name.rsplit('/', 1)[1]
    f1, f2, f3, f4 = file.rsplit('_')

    if f1.endswith('A') or f1.endswith('B'):
        f1 = f1[:-1]
    cell_name = f4.split('.', 1)[0]
    if cell_name.startswith('A') or cell_name.startswith('B'):
        cell_name = cell_name[1:]
    if cell_name.startswith('x'):
        feature_list.append(f1 + cell_name)
    else:
        feature_list.append(f1 + 'x' + cell_name)

    return w_percentage_list, feature_list


def run_feature_generation(args):
    windowsize_list = args.window_size
    windowsize_list.sort(reverse=True)

    path = args.file
    output_file = args.output_file
    jobs = 1

    if args.jobs:
        jobs = args.jobs

    file_name, ending = output_file.rsplit('.', 1)
    distribution_file = open(file_name + '_window_distribution.' + ending, 'w')
    output = open(output_file, 'w')
    features = get_header(windowsize_list)
    output.write('\t'.join(features))
    output.write('\n')

    if os.path.isfile(path):
        bamfile_name = path
        w_list, feature_list = get_bam_characteristics(jobs, windowsize_list, bamfile_name)
        distribution_file.write("\t".join(w_list))
        distribution_file.write('\n')
        output.write("\t".join(feature_list))
        output.write('\n')

    else:
        extension = '.bam'
        if args.bam_extension is not None:
            extension = args.bam_extension
        for path, subdirs, files in os.walk(path):
            for name in files:
                if name.endswith(extension):
                    next_cell = os.path.join(path, name)
                    bamfile_name = next_cell
                    w_list, feature_list = get_bam_characteristics(jobs, windowsize_list, bamfile_name)
                    distribution_file.write("\t".join(w_list))
                    distribution_file.write('\n')
                    output.write("\t".join(feature_list))
                    output.write('\n')
