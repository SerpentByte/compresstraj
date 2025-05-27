import warnings
warnings.filterwarnings("ignore")

import numpy as np
from MDAnalysis import Writer, Universe
import pickle
from tqdm.auto import trange, tqdm
import json 
import argparse
import os

parser = argparse.ArgumentParser(description="Recomposed fragmented decompressed trajectories.")
parser.add_argument('-r', '--reffile', type=str, required=True, help='Path to the reference file (pdb/gro)')
parser.add_argument('-o', '--outdir', type=str, help='output directory', default=".")
parser.add_argument('-c', '--config', type=str, help='config JSON', required=True)
parser.add_argument('-t', '--trajfile', type=str, help='Path to the trajectory file 1 (xtc)')
parser.add_argument('-sel', '--selection', type=str, default="all",
                    help="atom selections.")
args = parser.parse_args()

reffile = args.reffile
outdir = args.outdir
trajfile = args.trajfile
selection = args.selection

config = json.load(open(args.config, "r"))

selections = config["selections"]

ref = Universe(reffile, reffile)
select = ref.select_atoms(" or ".join(selections))

pdbs = config["pdbs"]
xtcs = config["xtcs"]

univs = [Universe(p, x) for p, x in zip(pdbs, xtcs)]
lengths = np.array([len(u.trajectory) for u in univs])

assert np.prod(lengths==lengths[0]), "ERROR: all trajectories do not have the same length."

prefix = config["prefix"]

with Writer(f"{outdir}/{prefix}_decompressed.xtc", "w") as w:
    ts = ref.trajectory.ts
    for frame in trange(lengths[0]):
        positions = np.concatenate([u.trajectory[frame]._pos for u in univs])
        select.positions = np.copy(positions)
        w.write(select)

with Writer(f"{outdir}/{prefix}_select.pdb", "w") as w:
    w.write(select)

print(f"""recomposed trajectory written to {outdir}/{prefix}_decompressed.xtc
the respective pdb has been written to {outdir}/{prefix}_select.pdb""")    

if trajfile is None:
    exit(0)
    
ref = Universe(reffile)
traj1 = Universe(reffile, trajfile)

with Writer("temp.pdb", "w") as w:
    w.write(ref.select_atoms(selection))
traj2 = Universe("temp.pdb", f"{outdir}/{prefix}_decompressed.xtc")

assert len(traj1.trajectory) == len(traj2.trajectory), "Trajectories must be of the same length."

# calcaulating and saving RMSD
print("calculating RMSD between original and decompressed trajectory.", flush=True)

rmsd = []
for t1, t2 in tqdm(zip(traj1.trajectory[:len(traj2.trajectory)], traj2.trajectory), desc="Computing RMSD", total=len(traj2.trajectory)):
    pos1 = traj1.select_atoms(selection).positions - traj1.select_atoms(selection).center_of_geometry()
    pos2 = traj2.select_atoms(selection).positions - traj2.select_atoms(selection).center_of_geometry()
    rmsd.append(np.mean((pos1 - pos2)**2))

rmsd = np.sqrt(rmsd)*0.1 # to nm

np.savetxt(f"{outdir}/{prefix}_rmsd.txt", rmsd, fmt="%.3f", header="RMSD in nm")
print(f"RMSD values saved in {outdir}/{prefix}_rmsd.txt", flush=True)

print(f"RMSD = ({rmsd.min():.2f}, {rmsd.max():.2f}) nm", flush=True)

os.system("rm -rf temp.pdb")
