
import matplotlib.pyplot as plt
import pandas as pd


def add_plotting_parser(subparsers):
    parser = subparsers.add_parser('plot', help='create plots for produced data')
    parser.add_argument('--w_percentage', '-w', help='file with Watson percentage for each considered window',
                        required=False)
    parser.add_argument('--annotation', '-a', help='annotation file for comparing predictions', required=False)
    parser.add_argument('--probabilities', '-p', help='file with prediction probabilities', required=False)
    parser.add_argument('--output_file', '-o', help='name of output file', required=True)

    parser.set_defaults(execute=run_plotting)

    return subparsers


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
        plt.hist(class_0, bins=40, alpha=0.8, label='Class 0')
        plt.hist(class_1, bins=40, alpha=0.8, label='Class 1')
        plt.legend(loc='upper left')

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

    return
