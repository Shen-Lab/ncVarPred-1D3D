# Multimodal learning of noncoding variant effects using genome sequence and chromatin structure
Integrate 1D genome sequence and 3D chromatin structure for noncoding genome variant effect prediction

## Environment preparation
You may create conda environment needed, by running "conda env create -f ncvarpred_1d3d.yml".

## Data Download
Most data needed, except some sanity check needed procssed Hi-C derived normalized interaction frequency matrix, can be downloaded from the [link](). The most recent (Sei)[https://www.nature.com/articles/s41588-022-01102-2] based pretrained models can be downloaded from the [link]()

## Model training
### (DeepSEA)[https://www.nature.com/articles/nmeth.3547] based or (DanQ)[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914104/] based models can be trained by running the code train_model_mlp.py or train_model_gcn.py for our MLP or GCN models respectively. 

### GNN based architecture
Similar to the MLP based architecture, you may check the run_model_train.sh for more details. The DNABERT embedded whole genome vector can be downloaded [here](https://drive.google.com/file/d/1mC05WLGeNNO-nBGaYnCp4tw1Wkh6C9jd/view?usp=sharing).

## Model assessment (and extract motif)
After the best models are selected based on validation loss, you may interest in the prediction auROC and auPRC on the test set. At the same time, you may get the proposed motif based on the SOTA's sequence only model, our model with MLP and GCN for motif analysis.

## SNP/eQTL effect analysis
For noncoding mutation effection prediction, two examples are provided in this section.

### GTEx (cell line specific) eQTLs
You may need to download the data file [Cells_Transformed_fibroblasts.v7.egenes.txt for GM12878 and Cells_EBV-transformed_lymphocytes.v7.egenes.txt for IMR90](https://www.gtexportal.org/home/datasets). Then you may preprocess the downloaded file and use our model (or SOTA) to predict the epigenetic events profile changes.

### Pathogenic prediction
The code used to train the pathogenic prediction, CNN+MLP or CNN/RNN+GCN are provided.



## Citation:
If you find the code useful for your research, please consider citing our paper:
```
@article{Tan2022ncVar,
	author = {Tan, Wuwei and Shen, Yang},
	title = {Multimodal learning of noncoding variant effects using genome sequence and chromatin structure},
	elocation-id = {2022.12.20.521331},
	year = {2022},
	doi = {10.1101/2022.12.20.521331},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2022.12.20.521331v1},
	journal = {bioRxiv}
}
```


