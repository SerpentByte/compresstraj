from MDAnalysis import Universe
from MDAnalysis.analysis import align
import argparse
import numpy as np
from tqdm.auto import tqdm
import warnings 

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Process input files and model files for compression.")
parser.add_argument('-r1', '--reffile1', type=str, required=True, help='Path to the reference file (pdb/gro)')
parser.add_argument('-t1', '--trajfile1', type=str, required=True, help='Path to the trajectory file 1 (xtc)')
parser.add_argument('-r2', '--reffile2', type=str, required=True, help='Path to the reference file (pdb/gro)')
parser.add_argument('-t2', '--trajfile2', type=str, required=True, help='Path to the trajectory file 2 (xtc)')
parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output (txt)')

args = parser.parse_args()

reffile1 = args.reffile1
reffile2 = args.reffile2
trajfile1 = args.trajfile1
trajfile2 = args.trajfile2
output = args.output

ref = Universe(reffile1)
traj1 = Universe(reffile1, trajfile1)
traj2 = Universe(reffile2, trajfile2)

fit1 = align.AlignTraj(traj1, ref, select="not name H*")
fit1.run()

fit2 = align.AlignTraj(traj2, ref, select="not name H*")
fit2.run()

assert len(traj1.trajectory) == len(traj2.trajectory), "Trajectories must be of the same length."

rmsd = []
for t1, t2 in tqdm(zip(traj1.trajectory, traj2.trajectory), desc="Computing RMSD", total=len(traj1.trajectory)):
    pos1 = traj1.select_atoms("not name H*").positions - traj1.select_atoms("not name H*").center_of_mass()
    pos2 = traj2.select_atoms("not name H*").positions - traj2.select_atoms("not name H*").center_of_mass()
    rmsd.append(np.mean((pos1 - pos2)**2))

rmsd = np.sqrt(rmsd)*0.1 # to nm

np.savetxt(output, rmsd, fmt="%.3f", header="RMSD in nm")

print(f"RMSD = ({rmsd.min():.2f}, {rmsd.max():.2f}) nm")