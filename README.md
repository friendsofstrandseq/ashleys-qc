# ashleys-qc
Automated Selection of High quality Libraries for the Extensive analYsis of Strandseq data (ASHLEYS)

ASHLEYS is developed on Linux environments using Python3.7.
For a full working example on how to use ASHLEYS, please take a look at the [processing pipeline](https://github.com/friendsofstrandseq/ashleys-qc-pipeline).
Please note that the preprocessing steps in this pipeline, e.g. short-read alignment and read duplicate marking, are always
required to prepare suitable input files for ASHLEYS; the pipeline (code) itself, however, is just an example implementation, and not
*per se* part of ASHLEYS.

## Setup
Clone the repository via
``` python
git clone https://github.com/friendsofstrandseq/ashleys-qc.git ashleys-qc
cd ashleys-qc
```
Then create and activate the conda environment:
``` python
conda env create -f environment/ashleys_env.yml
conda activate ashleys
```
For final setup, run
 ``` python
python setup.py install
```
Now you should be able to see all possible modules with
``` python
./bin/ashleys.py --help
```

## Build status

Develop branch:

[![Build Status](https://travis-ci.org/friendsofstrandseq/ashleys-qc.svg?branch=develop)](https://travis-ci.org/friendsofstrandseq/ashleys-qc)

Master branch:

[![Build Status](https://travis-ci.org/friendsofstrandseq/ashleys-qc.svg?branch=master)](https://travis-ci.org/friendsofstrandseq/ashleys-qc)

## Feature Generation
Compute features for one or more BAM files for a given window size. For a detailed explanation
of what features are computed, please refer to the [feature documentation](docs/Features.md).

Example usage generating all necessary features for using the pretrained models for all
.bam files in the specified directory:
``` python
./bin/ashleys.py -j 23 features -f [folder_with_bamfiles] -w 5000000 2000000 1000000 \
 800000 600000 400000 200000 -o [feature_table.tsv]
```

## Model Training
Train a new classification model based on an annotation file specifying class 1 cells.
The model is trained with support vector classification based on grid search on hyperparamters. <br>
Example usage:
``` python
./bin/ashleys.py train -p [feature_table.tsv] -a [annotation.txt] -o [output.tsv]
```

## Prediction
Predict the class probabilities for new cells based on pre-trained models or based on customized models. <br>
The default model trained with support vector classification should identify low-quality cells of new data with high confidence. 
For detailed information about the generated files, please refer to the [output interpretation](docs/Output.md). 

Example usage for prediction based on this pretrained model:
``` python
./bin/ashleys.py predict -p [feature_table.tsv] -o [output_folder] -m models/svc_default.pkl
```
When using the pretrained models, it is necessary to have `scikit-learn v.0.23.2` installed, as the models were generated with this version. 
For customized models also a newer version of `scikit-learn` can be used.

## Plotting
Plot the distribution of prediction probabilities. <br>
Example usage:
``` python
./bin/ashleys.py plot -p [output_folder]/prediction.tsv -o [output_plot]
```

## Test Data
Example of test data prediction which directly compares the predicted class to the true annotation:
``` python
./bin/ashleys.py predict -p data/test_features.tsv -o prediction.tsv \
-m models/svc_default.pkl -a data/test_annotation.txt
```
