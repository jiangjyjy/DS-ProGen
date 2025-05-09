# DS-ProGen

DS-ProGen: A Dual-Structure Deep Language Model for Functional Protein Design

## Brief Introduction

Inverse Protein Folding (IPF) is a critical subtask in the field of protein design, aiming to engineer amino acid sequences capable of folding correctly into a specified three-dimensional (3D) conformation. Although substantial progress has been achieved in recent years, existing methods generally rely on either backbone coordinates or molecular surface features alone, which restricts their ability to fully capture the complex chemical and geometric constraints necessary for precise sequence prediction. To address this limitation, we present **DS-ProGen**, a dual-structure deep language model for functional protein design, which integrates both backbone geometry and surface-level representations. By incorporating backbone coordinates as well as surface chemical and geometric descriptors into a next-amino-acid prediction paradigm, DS-ProGen is able to generate functionally relevant and structurally stable sequences while satisfying both global and local conformational constraints. On the PRIDE dataset, DS-ProGen attains the current state-of-the-art recovery rate of **61.47%**, demonstrating the synergistic advantage of multi-modal structural encoding in protein design. Furthermore, DS-ProGen excels in predicting interactions with a variety of biological partners, including ligands, ions, and RNA, confirming its robust functional retention capabilities.

## Setup Environment


```bash
# Create and activate conda environment
conda create -n mmp python=3.9
conda activate mmp

# Install from setup file
pip3 install -r requirements.txt

# Install PyG
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# for surface data processing
conda install msms

```

## Inverse Folding
### Data Processing
```bash
cd inverse_folding/data
```


#### Data Source

Pretrain data is from [AFDB](https://www.alphafold.ebi.ac.uk/) and [ESM Metagenomic Atlas](https://esmatlas.com/), which can be found in a compressed format on [Foldcomp Databases](https://foldcomp.steineggerlab.workers.dev/).

Other data is sourced from the [PRIDE Benchmark Protein Design](https://github.com/chq1155/PRIDE_Benchmark_ProteinDesign). Please refer to the repository for detailed information on the dataset structure and content.
***
#### Download data

#### `download_data.py`

This script is responsible for downloading complete file data in the data list. 

***
#### Backbone Data preparation
```bash
cd backbone
```
#### `train_data_prepare.py, test_data_prepare.py`

After getting all data file, run this to process raw data to the form for training and reference.

Notice that it's better if running on a device have gpu as the coords should be encoded in advance.

#### `afdb_data_prepare.py`

If the dataset is too large to save the coords feature, run this to save as `lmdb` format.
***
#### Surface Data preparation
```bash
cd surface
```
Run the following scripts in order to prepare surface data:

    1. extract_cath_fragment.py
    2. msms_get_surface.py
    3. get_surface_fasta.py
    4. prepare_surface.py


### How to Run
#### Fine-tuning
finetune on backbone data:
```bash
bash inverse_folding/scripts/finetune.sh
```
or finetune on surface data:
```bash
bash inverse_folding/scripts/finetune_surface.sh
```
#### Inference
```bash
bash inverse_folding/scripts/inference.sh
```
or
```bash
bash inverse_folding/scripts/inference_surface.sh
```
#### Sampling
```bash
bash inverse_folding/scripts/sample.sh
```
***
### Evaluation

Run `inverse_folding/inverse_folding/evaluate.py` or `inverse_folding/inverse_folding/evaluate_surface.py` to get the results.
