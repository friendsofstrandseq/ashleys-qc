

import pysam
from multiprocessing import Pool
from collections import Counter
import statistics
import matplotlib as plt


def add_features_parser(subparsers):
    parser = subparsers.add_parser('features', help='create features for bam files')
    parser.add_argument('--jobs', '-j', help="the number of jobs used to generate features", type=int)
    parser.add_argument('--file', '-f', help='the name of the bam file to analyze', required=True)
    parser.add_argument('--window_size', '-w', help='window size for feature generation', type=int, required=True)
    parser.add_argument('--output_plot', '-p', help='create plot showing wc feature distribution', required=False)
    parser.add_argument('--output_file', '-o', help='name of output file, should be .tsv', required=True)

    parser.set_defaults(execute=run_feature_generation)

    return subparsers


def print_statistics(list, all, output):
    if len(list) < 2:
        output.write('\t0\t0')
        if all:
            output.write('\t0\t0')
        return

    output.write('\t' + str(statistics.stdev(list)) + '\t' + str(statistics.mean(list)))
    if all:
        output.write('\t' + str(statistics.variance(list)) + '\t' + str(statistics.median(list)))


def plot_wc_distribution(w_percentage_list, output_file):
    plt.hist(w_percentage_list, bins=100)
    plt.xlabel('Watson reads percentage')
    plt.ylabel('count')
    plt.axvline(0.1, 0, 100, label='W10')

    title = 'Watson reads distribution over features'
    plt.title(title)
    plt.savefig(output_file + '.png')
    return


def get_wc_composition(total_window_collection_wc, total_window_collection, create_plot, output):
    # create 10 features for 10% steps of w-percentage in windows
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
            if create_plot:
                w_percentage_list.append(w_percentage)
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
        output.write('0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0.0\t0')
        return values

    for i in range(0, 100, 10):
        c = 'W' + str(i+10)
        output.write(str(wc_collection[c] / total) + '\t')

    output.write(str(total))

    return values, wc_difference, w_percentage_list


def get_read_features(chrom):
    global bamfile_name, windowsize
    mapq_threshold = 10

    count_collection = Counter([])
    with pysam.AlignmentFile(bamfile_name, "rb") as bamfile:
        chromosomes = dict(zip(bamfile.references, bamfile.lengths))
        length = chromosomes.get(chrom)
        stepsize = int(windowsize / 2)

        window_collection = Counter([])
        window_collection_wc = Counter([])
        window_collection_chrom_w = Counter([])
        neighbor_difference = []
        window_collection_chrom_w.update({str(chrom): 1})
        window_collection_chrom_w.update({str(chrom)+'W': 1})

        # count reads in each window of size stepsize
        for i in range(0, length, stepsize):
            s = str(chrom) + str(i)
            window_collection.update({s:1})
            window_collection_wc.update({s: 1})
            window_collection_wc.update({s+'W': 1})
            window_collection_wc.update({s+'C': 1})
            for read in bamfile.fetch(chrom, i, i+windowsize):
                window_collection.update({s:1})
                if read.is_unmapped:
                    if read.reference_start > i+stepsize or i == 0:
                        count_collection.update({'unmapped': 1})
                    continue
                # count all mapped reads
                if read.reference_start > i + stepsize or i == 0:
                    count_collection.update({'mapped': 1})
                if read.is_supplementary or read.is_secondary or read.is_qcfail:
                    if read.reference_start > i + stepsize or i == 0:
                        count_collection.update({'supplementary': 1})
                        window_collection_wc.update({s+'_supp': 1})
                    continue
                if read.is_duplicate:
                    if read.reference_start > i + stepsize or i == 0:
                        count_collection.update({'duplicate': 1})
                    continue
                if read.mapping_quality < mapq_threshold:
                    if read.reference_start > i + stepsize or i == 0:
                        count_collection.update({'mapping_quality': 1})
                    continue
                if read.is_read2:
                    if read.reference_start > i + stepsize or i == 0:
                        count_collection.update({'read2': 1})
                    continue
                if read.reference_start > i + stepsize or i == 0:
                    count_collection.update({'good': 1})

                window_collection_wc.update({s: 1})
                window_collection_chrom_w.update({str(chrom): 1})
                if read.is_reverse:
                    window_collection_wc.update({s+'W': 1})
                    window_collection_chrom_w.update({str(chrom)+'W': 1})
                else:
                    window_collection_wc.update({s+'C': 1})

            if not i == 0:
                last_window = str(chrom) + str(i-stepsize)
                diff = window_collection[last_window] - window_collection[s]
                neighbor_difference.append(diff)

    return chrom, count_collection, window_collection, window_collection_wc, neighbor_difference, window_collection_chrom_w


def get_bam_characteristics(create_plot, jobs, Id, output):
    # read a BAM file and return different features for windows of the chromosomes
    chromosome_list = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11',
                       'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21',
                       'chr22', 'chrX'] #, 'chrY']

    list = ['unmapped', 'mapped', 'supplementary', 'duplicate', 'mapping_quality', 'read2', 'good']

    # collections with different counts over all chromosomes
    total_count_collection = Counter([])
    total_window_collection = Counter([])
    total_window_collection_wc = Counter([])
    total_neighbor_difference = []
    total_chrom_collection = Counter([])

    p = Pool(jobs)
    result = p.map(get_read_features, chromosome_list)

    for r in result:
        # print(r)
        total_count_collection += r[1]
        total_window_collection += r[2]
        total_window_collection_wc += r[3]
        total_neighbor_difference += r[4]
        total_chrom_collection += r[5]

    values, wc_difference, w_percentage_list = get_wc_composition(total_window_collection_wc, total_window_collection,
                                                                  create_plot, output)
    # print_statistics(wc_difference, True)
    print_statistics(values, False, output)
    print_statistics(total_neighbor_difference, False, output)

    # absolute filtering feature values
    for i in list:
        output.write('\t' + str(total_count_collection[i]))

    # relative filtering feature values
    total_reads = total_count_collection['mapped'] + total_count_collection['unmapped']
    for i in list:
        output.write('\t' + str(total_count_collection[i]/total_reads))

    # add filename as sample+cell
    file = Id.rsplit('/', 1)[1]
    f1, f2, f3, f4 = file.rsplit('_')

    if f1.endswith('A') or f1.endswith('B'):
        f1 = f1[:-1]
    cell_name = f4.split('.', 1)[0]
    if cell_name.startswith('A') or cell_name.startswith('B'):
        cell_name = cell_name[1:]
    if cell_name.startswith('x'):
        output.write('\t' + f1 + cell_name + '\n')
    else:
        output.write('\t' + f1 + 'x' + cell_name + '\n')

    return w_percentage_list


def run_feature_generation(args):
    global bamfile_name, windowsize
    windowsize = args.window_size
    Id = args.file
    plot = False
    output_file = args.output_file
    output = open(output_file, 'w')
    jobs = 1

    if args.jobs:
        jobs = args.jobs
    if args.output_plot:
        plot = True
        output_plot = args.output_plot

    bamfile_name = Id
    w_list = get_bam_characteristics(plot, jobs, Id, output)
    if plot:
        plot_wc_distribution(w_list, output_plot)
    output.close()
