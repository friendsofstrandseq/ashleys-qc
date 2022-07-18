import pandas as pd
import pickle
import logging
import warnings


def add_prediction_parser(subparsers):
    parser = subparsers.add_parser('predict', help='predict class probabilities for new cells')
    parser.add_argument('--path', '-p', help='path to feature table of data that should be predicted', required=False)
    parser.add_argument('--output', '-o', help='folder for output file', required=True)
    parser.add_argument('--model', '-m', help='pkl model to use for prediction', required=False)

    parser_group = parser.add_argument_group('Evaluate prediction for given annotation')
    parser_group.add_argument('--annotation', '-a', help='path to folder with annotation files', required=False)

    parser_group_comp = parser.add_argument_group('Compare two predictions')
    parser_group_comp.add_argument('--prediction_1', '-p1', help='prediction file for first model', required=False)
    parser_group_comp.add_argument('--prediction_2', '-p2', help='prediction file for second model', required=False)

    parser.set_defaults(execute=run_prediction)

    return subparsers


def predict_model(model_name, features):
    with open(model_name, 'rb') as m:
        warn_message = ''
        original_warning = []
        # check for UserWarning of different sklearn versions
        with warnings.catch_warnings(record=True) as w:
            clf = pickle.load(m)
            if len(w) != 0:
                for message in w:
                    versions = str(message).split('version')
                    if len(versions) > 2 and issubclass(w[-1].category, UserWarning):
                        warn_message = 'You are using a different version of scikit-learn than the one used for ' \
                                       'training the classification model. This may lead to unexpected behavior.\n'
                        version_model = versions[1][:8]
                        version_installed = versions[2][:8]
                        warn_message += 'The model was trained with scikit-learn version ' + version_model + \
                                        'while you have version ' + version_installed + ' installed.\n' + \
                                        'We recommend installing the identical version of scikit-learn for operational safety. ' \
                                        'However, the ASHLEYS run will now proceed...\n'

                    else:
                        original_warning = w

        # print new (or old) warnings outside of catch:
        if warn_message:
            warnings.warn(warn_message)
        if original_warning:
            for i in range(len(original_warning)):
                warnings.warn(original_warning[i].message, original_warning[-1].category)

        prediction = clf.predict(features)
        probability = clf.predict_proba(features)[:, 1]
    return prediction, probability


def compare_prediction(prediction_1, prediction_2, annotation, output):
    dataset_1 = pd.read_csv(prediction_1, sep='\t')
    dataset_2 = pd.read_csv(prediction_2, sep='\t')
    names = dataset_1['cell'].values
    probabilities_1 = dataset_1['probability'].values
    probabilities_2 = dataset_2['probability'].values

    combined_prediction = []
    for p_high, p_ok in zip(probabilities_1, probabilities_2):
        if p_high < 0.5 and p_ok < 0.5:
            combined_prediction.append(0)
        elif p_high > 0.5:
            combined_prediction.append(1)
        else:
            combined_prediction.append(0.5)

    with open(output, 'w') as o:
        o.write('cell\tcombined_prediction\n')
        for n, p in zip(names, combined_prediction):
            o.write(str(n) + '\t' + str(p) + '\n')

    if annotation is not None:
        evaluate_prediction(combined_prediction, annotation, names, output, (0.3, 0.7))
    return


def evaluate_prediction(probability, annotation, names, output, critical_bound):
    class_list = []
    with open(annotation) as f:
        annotation_list = [line.rstrip() for line in f]

    for n in names:
        if n in annotation_list:
            class_list.append(1)
        else:
            class_list.append(0)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    fn_cells = []
    fp_cells = []
    fn_critical = []
    fp_critical = []
    for p, c, n in zip(probability, class_list, names):
        if c == 1:
            if p < 0.5:
                fn += 1
                fn_cells.append(n)
                if p > critical_bound[0]:
                    fn_critical.append(n)
            else:
                tp += 1
        else:
            if p < 0.5:
                tn += 1
            else:
                fp += 1
                fp_cells.append(n)
                if p < critical_bound[1]:
                    fp_critical.append(n)

    with open(output[0] + '_accuracy.txt', 'w') as f:
        f.write('false positive predictions: ' + str(fp_cells) + '\n')
        f.write('false positive and critical predictions: ' + str(fp_critical) + '\n')
        f.write('false negative predictions: ' + str(fn_cells) + '\n')
        f.write('false negative and critical predictions: ' + str(fn_critical) + '\n')
        f.write('accuracy: ' + str((tp + tn)/(tp+tn+fp+fn)) + '\n')
        f.write('F1 score: ' + str((2*tp)/(2*tp + fp + fn)) + '\n')
        f.write('tp: ' + str(tp) + ', tn: ' + str(tn) + ', fp: ' + str(fp) + ', fn: ' + str(fn))
    return


def run_prediction(args):
    output = args.output
    file_name = output.rsplit('.', 1)
    log_file = file_name[0] + '.log'
    if args.logging is not None:
        log_file = args.logging

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.INFO, handlers=[logging.FileHandler(log_file)])

    if args.prediction_1 is not None and args.prediction_2 is not None:
        logging.info('comparing two predictions of different models')
        compare_prediction(args.prediction_1, args.prediction_2, args.annotation, file_name)
        return

    critical_bound = (0.3, 0.7)
    dataset = pd.read_csv(args.path, sep='\t')
    features = dataset.drop(columns=['sample_name'])
    names = dataset['sample_name'].values

    logging.info('predicting class probabilities, extracting critical predictions between bound ' + str(critical_bound))
    prediction, probability = predict_model(args.model, features)

    if args.annotation is not None:
        logging.info('comparing prediction to given annotation')
        evaluate_prediction(probability, args.annotation, names, file_name, critical_bound)

    pred_file = open(output, 'w')
    critical = open(file_name[0] + '_critical.tsv', 'w')
    pred_file.write('cell\tprediction\tprobability\n')
    critical.write('cell\tprobability\n')
    for i in range(len(names)):
        pred_file.write(names[i] + '\t' + str(prediction[i]) + '\t' + str(round(probability[i], 4)) + '\n')
        if critical_bound[0] < probability[i] < critical_bound[1]:
            critical.write(names[i] + '\t' + str(round(probability[i], 4)) + '\n')

    pred_file.close()
