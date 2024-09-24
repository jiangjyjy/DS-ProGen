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