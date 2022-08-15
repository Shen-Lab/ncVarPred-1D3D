# noncoding_genome_variant
Integrate genome sequence and protein structure for noncoding genome variant effect prediction

## Data preparation
Step 1 Prepare the interaction frequency matrixs (3D structure of the whole genome). You can run the preprocessing.py in the data_preprocessing folder. It may take a few hours. The interaction frequency matrix is normalized by runing the normalize_if.py. You need to download the [juicer tools](https://github.com/aidenlab/juicer/wiki/Download). The version 1.22.01 is used in this analysis. Due to the data availablility, only the cell line GM12878, IMR90 and K562's Hi-C experiment data are used in this analysis.

Step 2 Downlaod the data prepared by the [DeepSEA](http://deepsea.princeton.edu/help/). The training/validation/test dataset contains the 1K bp genome sequence, the genome coordinates (in hg19) and the corresponding labels of the 919 epigenetic events.

Step 3 For each genome sequence, find the matched region from the interaction frequency matrix. You may run the get_sequence_structure_matching_index.py. Each row of the  if_matrix_matching_index_(interaction frequency matrix resolution).txt is showing the start and the end coordiates of a region in the interaction frequency matrix.

Step 4 (optional) To reduce the memory burden during training and inference, the training and test set are splitted into multiple subsets. For each chromosome, (training: all except 8 and 9. testing: 8 and 9), each subset contain 10k samples.

## Sanity check (consistency among 1D gennome sequence, 3D chromatin structure and epigenetic profiles)

