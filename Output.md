## Output Interpretation
During prediction of new cells, several output files are generated in the specified output folder: <br>
`prediction.tsv` includes the predicted class assignment (0 or 1) together with the class 1 probability for each cell. Values close to 1 indicate high-quality cells, 
values close to 0 indicate low-quality cells. The decision boundary for the class assignment is 0.5. Values between 0.3 and 0.7 might need a second look to verify the class assignment 
and are again listed in `prediction_critical.tsv`. <br>
If an annotation file was specified to compare the predicted class assignments, 
`prediction_accuracy.tsv` provides performance information along with wrongly predicted cells.