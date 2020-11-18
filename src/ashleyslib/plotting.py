import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def add_plotting_parser(subparsers):
    parser = subparsers.add_parser('plot', help='create plots for produced data')

    parser.add_argument('--annotation', '-a', help='annotation file for comparing predictions')
    parser.add_argument('--probabilities', '-p', help='file with prediction probabilities')
    parser.add_argument('--output_file', '-o', help='name of output file', required=True)

    parser_group_comp = parser.add_argument_group('Plot the distribution of a small set of features')
    parser.add_argument('--feature_table', '-f', help='feature table of cells to plot')
    parser.add_argument('--feature_list', '-fl', help='list of features to plot')
    parser_group_comp.add_argument('--compare', '-c', help='plot feature list and compare it to those features')
    parser_group_comp.add_argument('--compare_annotation', '-ca', help='annotation file for comparing data')

    parser_group_w = parser.add_argument_group('Plot Watson distribution for ')
    parser_group_w.add_argument('--w_percentage', '-w', help='file with Watson percentage for each considered window')
    parser.set_defaults(execute=run_plotting)

    return subparsers


def plot_feature_range(feature_table, annotation, feature_list, output_file, compare, compare_annotation):
    features = pd.read_csv(feature_table, sep='\s+')
    alpha = 0.8
    if compare is not None:
        compare_features = pd.read_csv(compare, sep='\s+')

    if annotation is not None:
        with open(annotation) as f:
            annotation_list = [line.rstrip() for line in f]
        ones_hist_table = features[features['sample_name'].isin(annotation_list)]
        zeros_hist_table = features[~features['sample_name'].isin(annotation_list)]
        if compare_annotation is not None:
            with open(compare_annotation) as c:
                compare_annotation_list = [line.rstrip() for line in c]
            compare_ones = compare_features[compare_features['sample_name'].isin(compare_annotation_list)]
            compare_zeroes = compare_features[~compare_features['sample_name'].isin(compare_annotation_list)]

    feature_range = []
    for f in feature_list:
        values = features[f]
        feature_range.append((min(values), max(values)))

    plt.clf()
    fig, axs = plt.subplots(1, len(feature_list))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    axis = range(len(feature_list))
    for i in range(len(feature_list)):
        axs[axis[i]].set_xlim(0, 0.2)
        bin_list = np.arange(0, 0.2, 0.002)
        axs[axis[i]].set_ylim(0, 100)
        axs[axis[i]].spines['top'].set_visible(False)
        axs[axis[i]].spines['right'].set_visible(False)
        if annotation is not None:
            axs[axis[i]].hist(zeros_hist_table[feature_list[i]], alpha=alpha, bins=bin_list, label='hgsvc - Class 0',
                              color='#fdae61')
            axs[axis[i]].hist(ones_hist_table[feature_list[i]], alpha=alpha, bins=bin_list, label='hgsvc - Class 1',
                              color='#d7191c')
            if compare_annotation is not None:
                axs[axis[i]].hist(compare_ones[feature_list[i]], alpha=alpha, bins=bin_list,
                                  label='nbt19 - class 1', color='#2c7bb6')
                axs[axis[i]].hist(compare_zeroes[feature_list[i]], alpha=alpha, bins=bin_list,
                                  label='nbt19 - class 0', color='#abd9e9')
            elif compare is not None:
                axs[axis[i]].hist(compare_features[feature_list[i]], alpha=alpha, bins=bin_list, color='green',
                                  label='Prediction data')
        else:
            axs[axis[i]].hist(features[feature_list[i]], bins=bin_list)
            if compare is not None:
                axs[axis[i]].hist(compare_features[feature_list[i]], alpha=alpha, bins=bin_list, color='green',
                                  label='Prediction data')
        axs[axis[i]].set_title(feature_list[i])

    axs[0].legend(loc='upper right')

    for ax in axs.flat:
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

    fig.set_size_inches(16, 6)
    plt.savefig(output_file)
    return


def plot_wc_distribution(w_list, output_file):
    # plot percentage of Watson reads contained in all windows
    fig, ax = plt.subplots(1)
    ax.hist(w_list, bins=200)
    size = 20
    ax.set_xlabel('Watson reads percentage', fontsize=size)
    ax.set_ylabel('count', fontsize=size)
    ax.set_xlim(0, 1)
    ax.spines['top'].set_visible(False)

    lines = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ylim = 2000
    height = ylim
    ax.set_ylim(0, ylim)
    color_features = 'black'
    for l in lines:
        ax.axvline(l, 0, 100, color=color_features, linestyle='dashed')
        ax.text(l-0.07, height - 10, 'W' + str(int(l*100)), color=color_features, fontsize=size)

    ax.text(0.92, height - 10, 'W100', color=color_features, fontsize=size)
    fig = plt.gcf()
    fig.set_size_inches(16, 5)
    plt.savefig(output_file, dpi=200)
    return


def plot_prediction_hist(output_file, probability_file, annotation_file):
    # plot distribution of prediction probabilities
    dataframe = pd.read_csv(probability_file, sep='\t')
    probability = dataframe['probability'].values
    names = dataframe['cell'].values
    size = 10  # 15  # (size for paper plot)
    plt.clf()
    bins = np.linspace(0, 1, 50)
    if annotation_file is not None:
        # add colors for real class
        annotation = []
        with open(annotation_file) as f:
            annotation_list = [line.rstrip() for line in f]
        for n in names:
            if n in annotation_list:
                annotation.append(1)
            else:
                annotation.append(0)
        class_1 = []
        class_0 = []
        for a, p in zip(annotation, probability):
            if a == 0:
                class_0.append(p)
            else:
                class_1.append(p)

        alpha = 0.7
        plt.hist(class_0, bins=bins, alpha=alpha, label='Class 0')
        plt.hist(class_1, bins=bins, alpha=alpha, label='Class 1')
        plt.legend(loc='upper center')

    else:
        plt.hist(probability, bins=bins)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('class 1 probability', fontsize=size)
    plt.ylabel('count', fontsize=size)
    plt.savefig(output_file, bbox_inches='tight')

    return


def run_plotting(args):
    output = args.output_file
    if args.w_percentage is not None:
        dataframe = open(args.w_percentage, 'r')
        lines = dataframe.readlines()
        values = [161, 1576]
        for i in values:
            line = lines[i].replace('\n', '')
            w_percentage = line.split('\t')
            w_percentage = [float(x) for x in w_percentage]
            output_file = output[:(len(output)-4)] + '_' + str(i) + output[(len(output)-4):]
            plot_wc_distribution(w_percentage, output_file)
    if args.probabilities is not None:
        plot_prediction_hist(output, args.probabilities, args.annotation)
    if args.feature_table is not None:
        feature_list = ['W40_5.0mb', 'W70_5.0mb', 'W20_0.6mb', 'W90_0.6mb']  # , 'total_0.2mb']
        if args.feature_list is not None:
            feature_list = args.feature_list
        plot_feature_range(args.feature_table, args.annotation, feature_list, output, args.compare,
                           args.compare_annotation)

    return
