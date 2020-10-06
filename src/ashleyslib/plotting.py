
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ashleyslib.train_classification_model import get_relative_features


def add_plotting_parser(subparsers):
    parser = subparsers.add_parser('plot', help='create plots for produced data')
    parser.add_argument('--w_percentage', '-w', help='file with Watson percentage for each considered window',
                        required=False)
    parser.add_argument('--annotation', '-a', help='annotation file for comparing predictions', required=False)
    parser.add_argument('--probabilities', '-p', help='file with prediction probabilities', required=False)
    parser.add_argument('--feature_table', '-f', help='feature table of cells to plot', required=False)
    parser.add_argument('--feature_list', '-l', help='list of features to plot', required=False)
    parser.add_argument('--output_file', '-o', help='name of output file', required=True)
    parser.add_argument('--relative', dest='relative', action='store_true', default=False, required=False,
                        help='using only relative features')

    parser.set_defaults(execute=run_plotting)

    return subparsers


def plot_feature_range(feature_table, annotation, feature_list, output_file, relative):
    features = pd.read_csv(feature_table, sep='\s+')
    if relative:
        features = get_relative_features(features)

    if annotation is not None:
        with open(annotation) as f:
            annotation_list = [line.rstrip() for line in f]
        ones_hist_table = features[features['sample_name'].isin(annotation_list)]
        zeros_hist_table = features[~features['sample_name'].isin(annotation_list)]

    rows, cols = features.shape
    feature_range = []
    for f in feature_list:
        values = features[f]
        feature_range.append((min(values), max(values)))

    plt.clf()
    fig, axs = plt.subplots(1, len(feature_list))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    axis = range(len(feature_list))
    in_list = np.arange(0, 2.04, 0.04)
    for i in range(len(feature_list)):
        #axs[axis[i]].set_xlim(feature_range[i][0], feature_range[i][1])
        axs[axis[i]].set_xlim(0, max(1, feature_range[i][1]+0.04))

        bin_list = np.arange(0, max(1.04, feature_range[i][1] + 0.04), 0.04)
        #bin_list = np.arange(feature_range[i][0], feature_range[i][1], feature_range[i][1]/50)
        axs[axis[i]].set_ylim(0, rows/2)
        if annotation is not None:
            axs[axis[i]].hist(zeros_hist_table[feature_list[i]], alpha=0.8, bins=bin_list, label='Class 0')
            axs[axis[i]].hist(ones_hist_table[feature_list[i]], alpha=0.8, bins=bin_list, label='Class 1')
        else:
            axs[axis[i]].hist(features[feature_list[i]], bins=bin_list)
        axs[axis[i]].set_title(feature_list[i])

    axs[0].legend(loc='upper right')

    for ax in axs.flat:
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

    #plt.show()
    fig.set_size_inches(16, 6)
    plt.savefig(output_file)
    return


def plot_wc_distribution(w_percentage_list, output_file):
    # plot percentage of Watson reads contained in all windows
    dataframe = pd.read_csv(w_percentage_list, header=None, sep='\t')
    w_list = dataframe.values.tolist()[0]
    plt.hist(w_list, bins=200)
    plt.xlabel('Watson reads percentage')
    plt.ylabel('count')
    plt.xlim(0, 1)
    lines = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for l in lines:
        plt.axvline(l, 0, 100, label='W10', color='black')

    title = 'Watson reads distribution over features'
    plt.title(title)
    fig = plt.gcf()
    fig.set_size_inches(16, 4)
    plt.savefig(output_file, dpi=200)
    return


def plot_prediction_hist(output_file, probability_file, annotation_file):
    # plot distribution of prediction probabilities
    dataframe = pd.read_csv(probability_file, sep='\t')
    probability = dataframe['probability'].values
    names = dataframe['cell'].values

    plt.clf()
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
        bins = np.linspace(0, 1, 50)
        plt.hist(class_0, bins=bins, alpha=0.8, label='Class 0')
        plt.hist(class_1, bins=bins, alpha=0.8, label='Class 1')
        plt.legend(loc='upper center')

    else:
        plt.hist(probability, bins=30)

    plt.xlabel('class 1 probability')
    plt.ylabel('count')
    title = 'prediction distribution'
    plt.title(title)
    plt.savefig(output_file)

    return


def run_plotting(args):
    output = args.output_file
    if args.w_percentage is not None:
        plot_wc_distribution(args.w_percentage, output)
    if args.probabilities is not None:
        plot_prediction_hist(output, args.probabilities, args.annotation)
    if args.feature_table is not None:
        #feature_list = ['W40_5mb', 'W70_5mb', 'W20_0.6mb', 'W90_0.6mb', 'total_0.2mb']
        feature_list = ['W40_5.0mb', 'W70_5.0mb', 'W20_0.6mb', 'W90_0.6mb', 'total_0.2mb']
        if args.feature_list is not None:
            feature_list = args.feature_list
        plot_feature_range(args.feature_table, args.annotation, feature_list, output, args.relative)

    return
