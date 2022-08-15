# noncoding_genome_variant
Integrate genome sequence and protein structure for noncoding genome variant effect prediction

## Data preparation
Step 1 Prepare the interaction frequency matrixs (3D structure of the whole genome). You can run the preprocessing.py in the data_preprocessing folder. It may take a few hours. The interaction frequency matrix is normalized by runing the normalize_if.py. You need to download the [juicer tools](https://github.com/aidenlab/juicer/wiki/Download). The version 1.22.01 is used in this analysis. Due to the data availablility, only the cell line GM12878, IMR90 and K562's Hi-C experiment data are used in this analysis.

Step 2 Downlaod the data prepared by the [DeepSEA](http://deepsea.princeton.edu/help/). The training/validation/test dataset contains the 1K bp genome sequence, the genome coordinates (in hg19) and the corresponding labels of the 919 epigenetic events.

Step 3 For each genome sequence, find the matched region from the interaction frequency matrix. You may run the get_sequence_structure_matching_index.py. Each row of the  if_matrix_matching_index_(interaction frequency matrix resolution).txt is showing the start and the end coordiates of a region in the interaction frequency matrix.

Step 4 (optional) To reduce the memory burden during training and inference, the training and test set are splitted into multiple subsets. For each chromosome, (training: all except 8 and 9. testing: 8 and 9), each subset contain 10k samples.

## Sanity check (consistency among 1D gennome sequence, 3D chromatin structure and epigenetic profiles)

### Prior model training, focus on all samples prepared by the DeepSEA.
Step 1 Calculate the genome sequence similarity (1K bp, pairwise matching), epigenetic events profile similarity and the 3D structure similarity (interaction frequency).

Step 2 Use the one-sided KS test to check the significance of our assumption about consistency among 1D genome sequence, 3D chromatin structure and epigenetic profiles. You may run the step2_statistics_analysis_45degree_getsummary.py to get the summary among our 100 replicates.

### Post model training, focus the test set to check if we have learned the pattern and have better consistency in our prediction.
Step 3 & 4 similar to step 1 & 2, just make the one-sided KS test based on our prediction on the test set.

## Model training
### MLP based architecture

### GNN based architecture
