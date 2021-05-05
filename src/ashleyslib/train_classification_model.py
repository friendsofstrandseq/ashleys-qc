from sklearn import svm
from sklearn import ensemble
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np
import json
from time import time
import pickle
import logging


logger = logging.getLogger(__name__)


def add_training_parser(subparsers):
    parser = subparsers.add_parser("train", help="train new classification model")
    parser.add_argument(
        "--iterations",
        "-i",
        default=50,
        required=False,
        type=int,
        help="number of times new datasets are generated for model creation, initial: 50",
    )
    parser.add_argument("--path", "-p", help="path to feature table", required=True)
    parser.add_argument(
        "--annotation",
        "-a",
        help="path to annotation file containing class 1 cells",
        required=True,
    )
    parser.add_argument(
        "--features",
        "-f",
        required=False,
        type=int,
        help="number of features used from feature table, using all if not specified",
    )
    parser.add_argument("--output", "-o", help="name of output file", required=True)
    parser.add_argument(
        "--cv_runs",
        "-c",
        help="number of cv runs performed by grid search, initial: 5",
        required=False,
        default=5,
        type=int,
    )
    parser.add_argument(
        "--json", "-js", help="json file with the parameters for grid search", required=True
    )

    model_parser = parser.add_mutually_exclusive_group(required=False)
    model_parser.add_argument(
        "--svc",
        dest="classifier",
        action="store_true",
        help="running support vector classification",
    )
    model_parser.add_argument(
        "--gb", dest="classifier", action="store_false", help="running gradient boosting"
    )
    parser.set_defaults(classifier=True)
    parser.set_defaults(execute=run_model_training)

    return subparsers


# evaluate model performance by comparing expected output and true output
def evaluation(prediction, true_values, test, prediction_dataset, current_iteration):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    counter = 0
    wrong_ids = []
    names = test["sample_name"].values

    # create dataframe with overall prediction results:
    # if cell was part of test dataset: value equals prediction probability (for class 1)
    # if cell was part of training dataset: value is -1
    rows, cols = prediction_dataset.shape
    insert_column = [-1] * rows
    all_names = prediction_dataset["name"].values
    for n, i in zip(all_names, range(rows)):
        if n not in names:
            continue
        pred = prediction[list(names).index(n)]
        insert_column[i] = round(pred, 4)
    prediction_dataset["i" + str(current_iteration)] = insert_column

    zero_one_prediction = []
    for p in prediction:
        if p < 0.5:
            zero_one_prediction.append(0)
        else:
            zero_one_prediction.append(1)

    for p, t, s in zip(zero_one_prediction, true_values, names):
        counter += 1
        if p == 0 and t == 0:
            tn += 1
        elif p == 0 and t == 1:
            fn += 1
            wrong_ids.append(s)
        elif p == 1 and t == 1:
            tp += 1
        elif p == 1 and t == 0:
            fp += 1
            wrong_ids.append(s)

    return fp, fn, tp, tn, wrong_ids, prediction_dataset


# add first column to dataset, based on annotation file: all cells contained in file are labeled 1
def add_class_column(dataset, annotation):
    prediction_dataset = pd.DataFrame()
    names = dataset["sample_name"].values
    class_list = []

    for n in names:
        if n in annotation:
            class_list.append(1)
        else:
            class_list.append(0)

    dataset.insert(loc=0, column="class", value=class_list, allow_duplicates=True)
    prediction_dataset["name"] = names
    prediction_dataset["class"] = class_list
    return dataset, prediction_dataset


# split dataset into train and test data, with equal class size for training
def train_test_split(dataset, test_flag):
    y = dataset["class"].values
    size = len(dataset.index)
    df_0 = pd.DataFrame(columns=list(dataset.columns))
    df_1 = pd.DataFrame(columns=list(dataset.columns))

    for s in range(size):
        if y[s] == 0:
            df_0 = df_0.append(dataset[s : s + 1])
        if y[s] == 1:
            df_1 = df_1.append(dataset[s : s + 1])

    smaller_size = min(len(df_0.index), len(df_1.index))
    if test_flag:
        zeros = np.split(df_0, [int(smaller_size + 1)])
        ones = np.split(df_1, [int(smaller_size + 1)])

        train = pd.concat([zeros[0], ones[0]])
        test = []

    else:
        zeros = np.split(df_0, [int(((smaller_size + 1) / 4) * 3)])
        ones = np.split(df_1, [int(((smaller_size + 1) / 4) * 3)])

        train = pd.concat([zeros[0], ones[0]])
        test = pd.concat([zeros[1], ones[1]])
        test = test.sample(frac=1, random_state=2)

    train = train.sample(frac=1, random_state=2)

    return test, train


# write feature importance of current model to output file
def feature_importance(imp_file, iteration, imp_values):
    imp_file.write(str(iteration))
    for i in imp_values:
        imp_file.write("\t" + str(i))
    imp_file.write("\n")


# create model (gradient boosting or support vector)
def create_model(
    test,
    train,
    model,
    features,
    parameters,
    feature_imp_file,
    n,
    log_all_models,
    prediction_dataset,
    current_iteration,
    results,
    cv_runs,
    n_jobs,
):

    y_train = train["class"].values
    y_train = y_train.astype("int")
    y_test = test["class"].values

    x_train = train.iloc[:, 1 : features + 1].values
    x_test = test.iloc[:, 1 : features + 1].values
    samples_test = test.loc[:, "sample_name"].values

    if model == "gb":
        clf, feature_imp = create_gb(x_train, y_train, parameters, cv_runs, n_jobs)
        feature_importance(feature_imp_file, n, feature_imp)
        prediction = clf.predict_proba(x_test)[:, 1]
    else:
        clf, feature_imp = create_svc(x_train, y_train, parameters, cv_runs, n_jobs)
        feature_importance(feature_imp_file, n, feature_imp)
        prediction = clf.predict_proba(x_test)[:, 1]

    params = clf.best_params_

    fp, fn, tp, tn, wrong, prediction_dataset = evaluation(
        prediction, y_test, test, prediction_dataset, current_iteration
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    results[0] += accuracy
    results[1] += f1
    results[2] += tp
    results[3] += tn
    results[4] += fp
    results[5] += fn
    log_all_models.write(
        str(round(accuracy, 4))
        + "\t"
        + str(round(tp / (tp + fn), 4))
        + "\t"
        + str(round(tn / (tn + fp), 4))
        + "\t"
        + str(wrong)
        + "\n"
    )

    logger.info("best parameter combination: " + str(params))
    logger.info("with accuracy: " + str(accuracy) + " and F1 score: " + str(f1))
    logger.info(
        "sensitivity: {}, specificity: {}, precision: {}".format(
            tp / (tp + fn), tn / (tn + fp), tp / (tp + fp)
        )
    )
    logger.info(
        "false positives: "
        + str(fp)
        + " false negatives: "
        + str(fn)
        + " true positives: "
        + str(tp)
        + " true negatives: "
        + str(tn)
        + "\n"
    )

    return wrong, samples_test, results, prediction_dataset


# performing grid search with support vector classifier based on specified parameters
def create_svc(x_train, y_train, parameters, cv_runs, n_jobs):
    svc = svm.SVC(random_state=2, probability=True)
    clf = GridSearchCV(svc, parameters, cv=cv_runs, n_jobs=n_jobs)
    clf.fit(x_train, y_train)
    feature_imp = clf.best_estimator_.coef_.ravel()

    return clf, feature_imp


# performing grid search with gradient boosting classifier based on specified parameters
def create_gb(x_train, y_train, parameters, cv_runs, n_jobs):
    gbc = ensemble.GradientBoostingClassifier(random_state=2)
    clf = GridSearchCV(gbc, parameters, cv=cv_runs, n_jobs=n_jobs)
    clf.fit(x_train, y_train)
    feature_imp = clf.best_estimator_.feature_importances_

    return clf, feature_imp


# create an output file listing all samples that were wrongly predicted
def outfile_wrong_predictions(wrong, samples, output_file):
    dict_wrong = dict()
    samples_tested = dict()
    for w_list in wrong:
        for w in w_list:
            if w in dict_wrong:
                dict_wrong[w] = dict_wrong[w] + 1
            else:
                dict_wrong[w] = 1

    for s_list in samples:
        for s in s_list:
            if s in samples_tested:
                samples_tested[s] = samples_tested[s] + 1
            else:
                samples_tested[s] = 1

    output_file.write("samples\twrong_predictions\ttested\tpercentage\n")

    for key, value in sorted(dict_wrong.items()):
        output_file.write(
            str(key)
            + "\t"
            + str(value)
            + "\t"
            + str(samples_tested[key])
            + "\t"
            + str(round(value / samples_tested[key] * 100, 2))
            + "\n"
        )

    return


def run_model_training(args):
    start_time = time()
    num = args.iterations
    features = args.features
    svc_model = args.classifier
    test_flag = False

    output_file = open(args.output, "w")
    log_name = args.output.split(".tsv")

    with open(args.annotation) as f:
        annotation = [line.rstrip() for line in f]

    # load json file with hyperparameters
    with open(args.json, "r") as jfile:
        params = json.load(jfile)

    dataset = pd.read_csv(args.path, sep="\t", header=0)
    # sort dataframe for reproducible results
    dataset.sort_values(by=["sample_name"], inplace=True)

    feature_names = dataset.columns
    dataset, prediction_dataset = add_class_column(dataset, annotation)

    if features is None:
        features = dataset.shape[1] - 2

    log_all_models = open(log_name[0] + "_model_log.tsv", "w")
    log_feature_imp = open(log_name[0] + "_feature_imp.tsv", "w")
    wrong_predictions = []
    samples_tested = []

    logger.info("Input: " + str(args.path))
    logger.info("used parameters: " + str(params))
    log_all_models.write("accuracy\tsensitivity\tspecificity\twrong_predicted\n")
    feature_importance(log_feature_imp, "iteration", feature_names[:-1])

    if svc_model:
        logger.info("running " + str(num) + " iterations creating support vector classifiers\n")
    else:
        logger.info("running " + str(num) + " iterations creating gradient boosting classifiers\n")

    total_results = [0.0] * 6  # total values for accuracy, f1, tp, tn, fp, fn
    sample_set = dataset

    for n in range(num):
        logger.info("current iteration: " + str(n))
        current_iteration = n
        sample_set = sample_set.sample(frac=1, random_state=2)
        test, train = train_test_split(sample_set, test_flag)

        if svc_model:
            wrong, samples, total_results, prediction_dataset = create_model(
                test,
                train,
                "svc",
                features,
                params,
                log_feature_imp,
                n,
                log_all_models,
                prediction_dataset,
                current_iteration,
                total_results,
                args.cv_runs,
                args.jobs,
            )
        else:
            wrong, samples, total_results, prediction_dataset = create_model(
                test,
                train,
                "gb",
                features,
                params,
                log_feature_imp,
                n,
                log_all_models,
                prediction_dataset,
                current_iteration,
                total_results,
                args.cv_runs,
                args.jobs,
            )
        wrong_predictions.append(wrong)
        samples_tested.append(samples)

    y = dataset["class"].values
    x = dataset.iloc[:, 1 : features + 1].values
    if svc_model:
        final_clf, feature_imp = create_svc(x, y, params, args.cv_runs, args.jobs)
        feature_importance(log_feature_imp, "final", feature_imp)
    else:
        final_clf, feature_imp = create_gb(x, y, params, args.cv_runs, args.jobs)
        feature_importance(log_feature_imp, "final", feature_imp)
    logger.info("Final model\nparameter selection: {}".format(final_clf.best_params_))

    # save final model
    with open(log_name[0] + ".pkl", "wb") as f:
        pickle.dump(final_clf, f)

    prediction_dataset.to_csv(log_name[0] + "_prediction.tsv", sep="\t", index=False)

    outfile_wrong_predictions(wrong_predictions, samples_tested, output_file)

    if num > 0:
        logger.info("mean accuracy: " + str(total_results[0] / num))
        logger.info("mean F1 score: " + str(total_results[1] / num))
        logger.info(
            "mean tp: "
            + str(int(round(total_results[2] / num)))
            + ", tn: "
            + str(int(round(total_results[3] / num)))
            + ", fp: "
            + str(int(round(total_results[4] / num)))
            + ", fn: "
            + str(int(round(total_results[5] / num)))
            + "\n"
        )

    end_time = time()
    logger.info("time needed for model creation and prediction: " + str(end_time - start_time))

    output_file.close()
    log_all_models.close()
    return
