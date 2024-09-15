# Employing Artificial Neural Networks for Optimal Storage and Facile Sharing of Molecular Dynamics Simulation Trajectories

This package provides a set of functions and classes to process biomolecular simulation trajectories and implement autoencoders for dimensionality reduction and reconstruction of trajectory data. The package uses PyTorch, PyTorch Lightning, and MDAnalysis for these purposes.

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
python -c "import compressTraj as ct; print(ct.__version__)"
``````

## Usage
Example scripts are present in the scripts directory.
The scripts can be used without any modification. 
Each script needs to be fed information using flags.
Each script has its own documentation regarding what to 
pass using the flags.</br>
Use `python <script> -h` to get the documentation regarding the flags 
for that script.
