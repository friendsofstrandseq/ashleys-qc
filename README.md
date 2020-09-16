# ashleys-qc
Automated Selection of High quality Libraries for the Extensive analYsis of Strandseq data (ASHLEYS)

## Feature Generation
Create features for bam files for a given window size

## Model Training
Train a new classification model based on an annotation file specifying class 1 cells.
The model is trained with gradient boosting based on grid search on hyperparamters.

## Prediction
Predict the class probabilities for new cells based on pre-trained models or based on customized models.
