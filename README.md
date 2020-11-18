# ashleys-qc
Automated Selection of High quality Libraries for the Extensive analYsis of Strandseq data (ASHLEYS)

ASHLEYS is developed on Linux environments using Python3.6.
For automatic installation and data prediction use the [pipeline](https://github.com/friendsofstrandseq/ashleys-qc-pipeline).

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
python setup.py develop
```
Now you should be able to see all possible modules with 
``` python
./bin/ashleys.py --help
```

## Feature Generation
Create features for bam files for a given window size. <br>
Example usage generating all necessary features for using the pretrained models for all 
.bam files in a specified directory : 
``` python
./bin/ashleys.py features -f [folder_with_bamfiles] -w 5000000 2000000 1000000 \
 800000 600000 400000 200000 -j 23 -o [feature_table.tsv] 
```

## Model Training
Train a new classification model based on an annotation file specifying class 1 cells.
The model is trained with support vector classification based on grid search on hyperparamters. <br>
Example usage: 
``` python
./bin/ashleys.py train -p [feature_table.tsv] -a [annotation.txt] -o [output.tsv] -j models/dict_svc.json
```

## Prediction
Predict the class probabilities for new cells based on pre-trained models or based on customized models. <br>
Example usage for prediction based on high quality pretrained model: 
``` python
./bin/ashleys.py predict -p [feature_table.tsv] -o [output_folder] -m models/hgsvc_high-qual.pkl
```

## Plotting
Plot the distribution of prediction probabilities. <br>
Example usage: 
``` python
./bin/ashleys.py plot -p [output_folder]/prediction_probabilities.tsv -o [output_plot]
```