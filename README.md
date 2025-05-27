# compresstraj

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

```
git clone https://github.com/SerpentByte/compresstraj.git
```

Then create a new environment using the provided `environment.yaml` file:</br>
```
cd compresstraj
```

```
conda env create -f environment.yaml
```

activate the environment using</br>
```
conda activate compresstraj
```

You can then install the package via `pip`:

```
pip install -e .
```



---

## Examples
Please run these commands to test for successful installations.</br>
These also serve as usage examples.

### protein
For compression:</br>
Go into the path `examples/protein` and execute</br>

```
python ../../scripts/compress.py -r hewl.pdb -t hewl.xtc -p hewl -e 100 -b 32 -c 20 -sel "protein" -o test
```
</br>


For decompression:</br>

```
python ../../scripts/decompress.py -m test/hewl_model.pt -s test/hewl_scaler.pkl -r test/hewl_select.pdb -c test/hewl_compressed.pkl -cog test/hewl_cog.pkl -p hewl -sel "protein" -o test
```
</br>

### protein-ligand
For compression:</br>
Go into the path `examples/protein_ligand` and execute</br>

```
# protein
python ../../scripts/compress.py -r p450_cam.pdb -t p450_cam.xtc -p prt -e 100 -c 20 -sel "(not resname HEM) and (not resname CAM)" -o test
# HEME subunit
python ../../scripts/compress.py -r p450_cam.pdb -t p450_cam.xtc -p hem -e 100 -c 20 -sel "resname HEM" -o test
# camphor
python ../../scripts/compress.py -r p450_cam.pdb -t p450_cam.xtc -p cam -e 100 -c 20 -sel "resname CAM" -o test
```
</br>

For decompression:</br>

```
# protein
python ../../scripts/decompress.py -m test/prt_model.pt -s test/prt_scaler.pkl -r test/prt_select.pdb -c test/prt_compressed.pkl -p prt -sel "(not resname HEM) and (not resname CAM)" -cog test/prt_cog.pkl -o test
# HEME subunit
python ../../scripts/decompress.py -m test/hem_model.pt -s test/hem_scaler.pkl -r test/hem_select.pdb -c test/hem_compressed.pkl -p hem -sel "resname HEM" -cog test/hem_cog.pkl -o test
# camphor
python ../../scripts/decompress.py -m test/cam_model.pt -s test/cam_scaler.pkl -r test/cam_select.pdb -c test/cam_compressed.pkl -p cam -sel "resname CAM" -cog test/cam_cog.pkl -o test
# recomposition
python ../../scripts/recompose.py -r p450_cam.pdb -c recompose.json -t p450_cam.xtc
```

#### Notes
- For decompression, the `-t` flag allows you to provide the original trajectory. If supplied, the script will compute framewise RMSDs and report both the maximum and minimum values. This can help assess reconstruction quality.
- A `-t` flag is also present for the `recompose.py` script which serves the same purpose. 
- The code is designed to minimize memory usage. If you encounter an out-of-memory (OOM) error on GPU or excessive RAM usage on CPU, try reducing the batch size to 1 using the `-b` flag. This may indicate that the model is too large for your system's memory.
- The recommended number of epochs to train a model is at least 1000 for actual usage.
- To view all the options one can pass to a script, use `python <script-path>.py -h`.

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
