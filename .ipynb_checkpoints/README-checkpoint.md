# compressTraj

**Trajectory Compression Using Deep Autoencoders**

`compressTraj` is a Python package designed to compress and decompress molecular dynamics trajectories using a deep neural network 
architecture called "autoencoders". </br>
It is built using `MDAnalysis`, `PyTorch`, and `PyTorch Lightning`, enabling high-performance and GPU-accelerated workflows.</br>
However GPU-acceleration is not neccessary but recommended.

---

## Features

- Autoencoder-based compression of MD trajectories
- Flexible software with a focus on end-user ease of use
- Upto 86% or more reeduction in file sizes with near-accurate reconstruction

---

## Installation
First clone the repository:</br>
`git clone https://github.com/SerpentByte/compresstraj`

Then create a new environment using the provided `environment.yaml` file:
```
cd compresstraj
conda env create -f environment.yaml
```

You can then install the package via `pip`:

`pip install .` or `pip install -e .`



---

## Examples
Please run these commands to test for successful installations.</br>
These also serve as usage examples.

### protein
For compression:</br>
Go into the path `examples/protein` and execute</br>
```python ../../scripts/compress.py -r hewl.pdb -t hewl.xtc -p hewl -e 100 -b 32 -c 20 -sel "protein" -o test```</br>
For decompression:</br>
```python ../../scripts/decompress.py -m test/hewl_model.pt -s test/hewl_scaler.pkl -r test/hewl_select.pdb -c test/hewl_compressed.pkl -p hewl -sel "protein" -o test```

### protein-ligand
For compression:</br>
Go into the path `examples/protein_ligand` and execute</br>
```python ../../scripts/compress_prt_lig.py -r t4l_l99a_bnz.pdb -t t4l_l99a_bnz.xtc -p t4l_l99a_bnz -e 100 -c 20 -lig "resname BNZ" -sel "protein" -o test```</br>
For decompression:</br>
```python ../../scripts/decompress_prt_lig.py -mp test/t4l_l99a_bnz_prt_model.pt -ml test/t4l_l99a_bnz_lig_model.pt -sp test/t4l_l99a_bnz_prt_scaler.pkl -sl test/t4l_l99a_bnz_lig_scaler.pkl -r test/t4l_l99a_bnz_select.pdb -cp test/t4l_l99a_bnz_prt_compressed.pkl -cl test/t4l_l99a_bnz_lig_compressed.pkl -lcom test/t4l_l99a_bnz_lig_com.npy -p t4l_l99a_bnz -sel "protein" -lig "resname BNZ" -o test```

_for decompression, there is an option to feed the original trajectory using `-t` flag. If fed, the script will calculate framewise RMSDs and report the maximum and minimum RMSDs. This can be used to test reconstruction quality_

---

## Reference
If you find the software useful and use it for storing/sharing MD trajectories, please cite:</br>
```
@article{Wasim2024.09.15.613125,
  author = {Wasim, Abdul and Sch{\"a}fer, Lars V. and Mondal, Jagannath},
  title = {Employing Artificial Neural Networks for Optimal Storage and Facile Sharing of Molecular Dynamics Simulation Trajectories},
  elocation-id = {2024.09.15.613125},
  year = {2024},
  doi = {10.1101/2024.09.15.613125},
  publisher = {Cold Spring Harbor Laboratory},
  journal = {bioRxiv},
  url = {https://www.biorxiv.org/content/early/2024/09/19/2024.09.15.613125}
}
```

The pre-print is available at https://doi.org/10.1101/2024.09.15.613125