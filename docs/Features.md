## List of features calculated for each library during feature generation

* 7 features for number of filtered reads: 
\# unmapped reads, \# mapped reads, \# duplicates, \# reads with mapping quality < 10, 
\# read2 reads, # supplementary reads (including also secondary and qc failed reads), 
\# good reads. 
Each of those values is normalized by (\# unmapped + \# mapped)<br>
Those features are calculated once for each single-cell
* 10 features for Watson distribution, specifying the fraction of Watson reads per window
(W10, W20, W30, W40, W50, W60, W70, W80, W90, W100), calculated for each window size
* 1 feature for (total number of non-empty windows) / (total number of windows), calculated for each window size

* **optional**: With `--statistics`, eight additional features can be included. The four statistical values 
mean, median, standard deviation and variance  are calculated for the read count for 
each window as well as for the deviation in number of reads for neighboring windows. 
Those values are calculated for each window size.

When using the pre-trained models that are provided by ASHLEYS, the features should be generated for seven 
window sizes: 5, 2, 1, 0.8, 0.6, 0.4 and 0.2 Mbp.<br>
Note that the classification model has to be trained on the exact same set of features 
that is used for class prediction.