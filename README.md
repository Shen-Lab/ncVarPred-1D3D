# noncoding_genome_variant
Integrate genome sequence and protein structure for noncoding genome variant effect prediction

Step 1 Prepare the interaction frequency matrixs (3D structure of the whole genome). You can run the preprocessing.py in the data_preprocessing folder. It may take a few hours. You need to download the [juicer tools](https://github.com/aidenlab/juicer/wiki/Download). The version 1.22.01 is used in this analysis.

Step 2 Downlaod the data prepared by the [DeepSEA](http://deepsea.princeton.edu/help/). The training/validation/test dataset contains the 1K bp genome sequence, the genome coordinates (in hg19) and the corresponding labels of the 919 epigenetic events.

Step 3 For each genome sequence, find the matched region from the interaction frequency matrix.
