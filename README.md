# Multimodality ProteinLLM

## Setup Environment


```bash
# Create and activate conda environment
conda create -n mmp python=3.9
conda activate mmp

# Install from setup file
pip3 install -r requirements.txt

# Install PyG
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

```

## Inverse Folding
### Data Processing

#### Data Source

The dataset is sourced from the [PRIDE Benchmark Protein Design](https://github.com/chq1155/PRIDE_Benchmark_ProteinDesign). Please refer to the repository for detailed information on the dataset structure and content.
***
#### Download Extra data

#### `download_extra_data.py`

A relatively complete PDB file dataset is assumed has been downloaded. 

This script is responsible for downloading additional file data that is not included in the original file dataset but in the data list. 

If some files are missing, run this program to fill in the gaps.
***
#### Data preparation
#### `train_data_prepare.py, test_data_prepare.py`

After getting all data file, run this to process raw data to the form for training and reference.

Notice that it's better if running on a device have gpu as the coords should be encoded in advance.
***
### How to Run
#### Fine-tuning
```bash
bash inverse_folding/scripts/finetune.sh
```

#### Imference
```bash
bash inverse_folding/scripts/inference.sh
```

#### Sampling
```bash
bash inverse_folding/scripts/sample.sh
```
***
### Metrics

Run `inverse_folding/inverse_folding/metric.py` to get result.


