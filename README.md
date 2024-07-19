# Compression of Molecular Dynamics trajectories using AutoEncoders

This package provides a set of functions and classes to process molecular dynamics (MD) trajectories and implement autoencoders for dimensionality reduction and reconstruction of trajectory data. The package uses PyTorch, PyTorch Lightning, and MDAnalysis for these purposes.

## Dependencies
- numpy
- pickle
- tqdm
- cudatoolkit
- scikit-learn
- pytorch
- pytorch-lightning
- mdanalysis

## Installation

Clone the repository and navigate into the project directory:

```bash
git clone https://github.com/SerpentByte/compressTraj
cd compressTraj
conda create -n new_env python wheel setuptools -y
conda activate new_env
python setup.py sdist bdist_wheel
pip install dist/compressTraj-2024.0-py3-none-any.whl
python -m "import compressTraj as ct; print(ct.__version__)
``````

## Usage
Example scripts are present in the scripts directory.
The scripts can be used without any modification. 
For automated data preparation, training of autoencoders
and data compression please see autorun.sh in scripts.
autorun.sh also contains a part to decompress. please comment or 
uncomment the required parts for automated usage of the package
along with the scripts.
