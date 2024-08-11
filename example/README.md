## Usage
Example usage:</br>
```python ../scripts/compress.py -r example.pdb -t example.xtc -p temp -e 200 -sel "all" -l 16```</br>
This will process the trajectory `example.xtc` and train an autoencoder to compress the trajectory in a `16` dimensional latent space.</br>
To decompress, use</br>
```python ../scripts/decompress.py -m temp_model.pt -s temp_scaler.pkl -r example.pdb -t example.xtc -c temp_compressed.pkl -p temp -sel "all"```</br>
This will create `temp_decompressed.xtc` and `temp_select.pdb`. The xtc file is the decompressed trajectory. Since the original trajectories
are also fed, the default behaviour is to calculate the RMSD.
However, in a real use case, those trajectories will not be available during decompression. The script will just decompress and will not try
to calculate RMSD in such cases. The flag `-t` is hence optional. providing it switches on the RMSD calculation.
