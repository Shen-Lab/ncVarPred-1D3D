# Multimodal learning of noncoding variant effects using genome sequence and chromatin structure
Integrate 1D genome sequence and 3D chromatin structure for noncoding genome variant effect prediction

## Environment preparation
You may create conda environment needed, by running "conda env create -f ncvarpred_1d3d.yml".

## Data Download
Most data needed, except some sanity check needed procssed Hi-C derived normalized interaction frequency matrix, can be downloaded from the [Zenodo](). The most recent [Sei](https://www.nature.com/articles/s41588-022-01102-2) based pretrained models can be downloaded from the [Zenodo](). You need to decompress all files, by running the code "bash decompress.sh".

## Model training (all codes in the model_training folder)
[DeepSEA](https://www.nature.com/articles/nmeth.3547) based or [DanQ](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4914104/) based models can be trained by running the code train_model_mlp.py or train_model_gcn.py for our MLP or GCN models respectively. You may custom the learning rate, regularization. You may find more details in run_train_model.sh.

Sei based models can be trained by running the code train_sei_mlp.py or train_sei_gcn.py for our proposed MLP or GCN architecture respectively. You may custom the learning term and regularization.You may find more details in run_train_sei_mlp_gcn.sh. 

## Model assessment (all codes in the model_testing folder)
The test set performance of all pretrained model can be assessed by running the test_model_mlp.py or test_model_gcn.py for our proposed MLP or GCN architecture respectively. You may get all results by running the code run_test_model.sh. If you are interested the Sei related models, you may get the test set performance by running the code test_sei_mlp.py, test_sei_gcn.py. Or for simplicity sake, use the run_test_sei.sh. Please note that running that code to assess one pretrained Sei related mdoel on A100 GPU may take around 8 hours. The DeepSEA or DanQ related models may only tabke around 10 minutes.

## noncoding variant effect prediction (all codes in the ncVar_effect_prediction folder)
For noncoding mutation effection prediction, a eQTL effect prediction and a few-shot learning pathogenic variant prediction experiments are provided.

### GTEx (cell line specific) eQTLs
You may need to download the data file [Cells_Transformed_fibroblasts.v7.egenes.txt for GM12878 and Cells_EBV-transformed_lymphocytes.v7.egenes.txt for IMR90](https://www.gtexportal.org/home/datasets). Then you may preprocess the downloaded file and use our model (or SOTA) to predict the epigenetic events profile changes. More details about how the eQTL information are extracted, processed and preared as our model input can be found in the code "preprocess_GTEx_step1_collect_eqtl_list.py" and "prepare_seq.py". This may be helpful if you want to use our model for novel variants effect prediction.

### ncVarDB
Raw data is available at [ncVarDB repo](https://github.com/Gardner-BinfLab/ncVarDB)

### Our models' inference results
You may get all inference results by running the code "run_model_inference.sh". Then you may use our predicted epigenetic profile as input feature to train the ML model to predict the noncoding mutation effect.

### Few-shot learning for pathogenic variant prediction
The few shot learning pathogenic variant prediction model can be trained (and tested) by running the code "run_pathogenic_fewshot_learning.sh" and "run_pathogenic_fewshot_learning_test.sh".

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


