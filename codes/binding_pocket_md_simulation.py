# =============================================================
# Binding Pocket Detection using Fpocket
# =============================================================


from google.colab import drive
drive.mount('/content/drive')

import os
work_dir = "/content/drive/MyDrive/Rayca_eval/"
os.makedirs(work_dir, exist_ok=True)

!apt-get update
!apt-get install -y build-essential git

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/Discngine/fpocket.git
# %cd fpocket
!make

!./bin/fpocket -h

!mkdir -p /content/drive/MyDrive/Rayca_eval/fpockets

!./bin/fpocket -f /content/drive/MyDrive/Rayca_eval/Data/3bik_dock_prep.pdb
!mv 3bik_dock_prep_out /content/drive/MyDrive/Rayca_eval/fpockets/

!ls -lh /content/drive/MyDrive/Rayca_eval/Data/

!mkdir -p /content/drive/MyDrive/Rayca_eval/fpockets

!mv -v /content/drive/MyDrive/Rayca_eval/Data/3bik_dock_prep_out /content/drive/MyDrive/Rayca_eval/fpockets/

ls

!ls -lh /content/drive/MyDrive/Rayca_eval/fpockets/3bik_dock_prep_out/

!cat /content/drive/MyDrive/Rayca_eval/fpockets/3bik_dock_prep_out/3bik_dock_prep_info.txt

!grep "POCKET 2" -A 100 /content/drive/MyDrive/Rayca_eval/fpockets/3bik_dock_prep_out/3bik_dock_prep_out.pdb > /content/drive/MyDrive/Rayca_eval/fpockets/best_pocket.pdb

!head -20 /content/drive/MyDrive/Rayca_eval/fpockets/3bik_dock_prep_out/3bik_dock_prep_out.pdb

# =============================================================
# Molecular Dynamics Simulation Analysis using GROMACS and MDAnalysis
# =============================================================


"""Gromacs_md_simulation_mdAnalysis

"""

from google.colab import drive
drive.mount('/content/drive')

!apt-get update
!apt-get install -y gromacs gromacs-mpi

!gmx --version

!pip install MDAnalysis

!pip install MDAnalysis[analysis] freesasa

!mkdir -p /content/drive/MyDrive/Rayca_eval/MD_simulation


!echo 15 | gmx pdb2gmx -f /content/drive/MyDrive/Rayca_eval/Data/pocket2_residues_dock_prep.pdb \
-o best_pocket.gro -water spce -ignh

!gmx editconf -f best_pocket.gro -o pocket_box.gro -c -d 1.0 -bt cubic


# ; ions.mdp - Used for ion addition

# integrator               = steep
# emtol                    = 1000.0
# emstep                   = 0.01
# nsteps                   = 50000
# nstlist                  = 1
# cutoff-scheme            = Verlet
# ns_type                  = grid
# rlist                    = 1.0
# coulombtype              = PME
# rcoulomb                 = 1.0
# vdw-type                 = cutoff
# rvdw                     = 1.0
# pbc                      = xyz
#

!gmx solvate -cp pocket_box.gro -cs spc216.gro -o solvated_pocket.gro -p topol.top

!gmx grompp -f ions.mdp -c solvated_pocket.gro -p topol.top -o ions.tpr

!echo 13 | gmx genion -s ions.tpr -o neutralized_pocket.gro -p topol.top -pname NA -nname CL -neutral

mdp_content = """; Energy minimization parameters
integrator    = steep
emtol         = 1000.0
emstep        = 0.01
nsteps        = 50000
nstlist       = 1
cutoff-scheme = Verlet
ns_type       = grid
coulombtype   = PME
rcoulomb      = 1.0
rvdw          = 1.0
pbc           = xyz
"""





!gmx grompp -f /content/drive/MyDrive/Rayca_eval/MD_simulation/em.mdp \
           -c /content/drive/MyDrive/Rayca_eval/MD_simulation/neutralized_pocket.gro \
           -p /content/drive/MyDrive/Rayca_eval/MD_simulation/topol.top \
           -o /content/drive/MyDrive/Rayca_eval/MD_simulation/em.tpr

!gmx mdrun -v -deffnm /content/drive/MyDrive/Rayca_eval/MD_simulation/em

mdp_content = """; NVT Equilibration (Constant Volume and Temperature)
; This step stabilizes the system at 300K before proceeding to NPT equilibration.

; Algorithm for integration
integrator    = md        ; Molecular dynamics integrator

; Simulation steps
nsteps        = 50000     ; Run simulation for 50,000 steps
dt            = 0.002     ; Time step of 2 fs

; Output control
nstxout       = 1000      ; Save coordinates every 1000 steps
nstvout       = 1000      ; Save velocities every 1000 steps
nstenergy     = 1000      ; Save energy every 1000 steps
nstlog        = 1000      ; Write log every 1000 steps

; Constraints
continuation  = no        ; Start fresh, not continuing from a previous run
constraint_algorithm = lincs  ; Apply LINCS algorithm to constrain bonds
constraints   = all-bonds ; Constrain all bonds
lincs_iter    = 1         ; Number of iterations in LINCS
lincs_order   = 4         ; Maximum order of the expansion in LINCS

; Neighbor searching
cutoff-scheme = Verlet    ; Use Verlet cutoff scheme
ns_type       = grid      ; Grid-based neighbor searching
nstlist       = 10        ; Update neighbor list every 10 steps

; Electrostatics and van der Waals interactions
rcoulomb      = 1.0       ; Coulomb interaction cutoff distance (nm)
rvdw          = 1.0       ; Van der Waals interaction cutoff distance (nm)
coulombtype   = PME       ; Particle-Mesh Ewald (PME) for long-range electrostatics

; Temperature coupling (Thermostat)
tcoupl        = V-rescale ; Velocity rescaling thermostat
tc-grps       = System    ; Apply temperature coupling to the entire system
tau_t         = 0.1       ; Time constant for temperature coupling
ref_t         = 300       ; Reference temperature (K)

; Pressure coupling (disabled for NVT)
pcoupl        = no        ; No pressure coupling in NVT

; Periodic boundary conditions
pbc           = xyz       ; Use periodic boundary conditions in all directions
"""

file_path = "/content/drive/MyDrive/Rayca_eval/MD_simulation/nvt.mdp"

# Save the .mdp file
with open(file_path, "w") as f:
    f.write(mdp_content)

print(f"Saved nvt.mdp to {file_path}")

!gmx grompp -f /content/drive/MyDrive/Rayca_eval/MD_simulation/nvt.mdp -c neutralized_pocket.gro -r neutralized_pocket.gro -p topol.top -o nvt.tpr

!echo 1 | gmx mdrun -v -deffnm nvt

!gmx editconf -f step0c.pdb -o /content/drive/MyDrive/Rayca_eval/MD_simulation/water_only.pdb

!echo "Protein" | gmx select -f step0c.pdb -s nvt.tpr -on /content/drive/MyDrive/Rayca_eval/MD_simulation/bad_waters.ndx -select 'same residue as (name OW and within 0.2 of group "Protein")'

!echo "bad_waters" | gmx trjconv -s nvt.tpr -f step0c.pdb -n /content/drive/MyDrive/Rayca_eval/MD_simulation/bad_waters.ndx -o /content/drive/MyDrive/Rayca_eval/MD_simulation/bad_waters.pdb

!gmx editconf -f step0c.pdb -o cleaned_pocket.pdb -c -d 1.0

!gmx solvate -cp cleaned_pocket.pdb -cs spc216.gro -o solvated.gro -p topol.top

!gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr
!echo 13 | gmx genion -s ions.tpr -o neutralized_pocket.gro -p topol.top -pname NA -nname CL -neutral

!gmx grompp -f em.mdp -c neutralized_pocket.gro -p topol.top -o em.tpr
!gmx mdrun -v -deffnm em

!gmx grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr
!gmx mdrun -v -deffnm nvt

 
# ; NPT equilibration
# title                   = NPT Equilibration
# define                  = -DPOSRES  ; Position restrain the protein
# integrator              = md        ; Molecular dynamics algorithm
# dt                      = 0.002     ; Time step (ps)
# nsteps                  = 50000     ; 100 ps
# nstxout                 = 5000      ; Save coordinates every 10 ps
# nstvout                 = 5000      ; Save velocities every 10 ps
# nstenergy               = 1000      ; Save energies every 2 ps
# nstlog                  = 1000      ; Write log every 2 ps
# 
# ; Neighbor searching
# cutoff-scheme           = Verlet
# nstlist                 = 10
# rlist                   = 1.0
# rcoulomb                = 1.0
# rvdw                    = 1.0
# 
# ; Electrostatics
# coulombtype             = PME
# pme_order               = 4
# fourierspacing          = 0.16
# 
# ; Van der Waals interactions
# vdw-modifier            = force-switch
# rvdw-switch             = 0.9
# 
# ; Temperature coupling
# tcoupl                  = V-rescale
# tc-grps                 = Protein Non-Protein
# tau-t                   = 0.1  0.1
# ref-t                   = 300  300
# 
# ; Pressure coupling
# pcoupl                  = Berendsen
# pcoupltype              = isotropic
# tau-p                   = 2.0
# ref-p                   = 1.0
# compressibility         = 4.5e-5
# 
# ; Constraints (FIXED)
# constraints             = h-bonds
# constraint_algorithm    = LINCS
# lincs_iter              = 1
# lincs_order             = 4
#

!gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -p topol.top -o npt.tpr -maxwarn 1

!gmx mdrun -deffnm npt -v


# ; Production MD run for 100 ns
# 
# integrator              = md
# dt                      = 0.002
# nsteps                  = 50000000    ; 100 ns
# nstxout                 = 50000       ; Save coordinates every 100 ps
# nstvout                 = 50000       ; Save velocities every 100 ps
# nstenergy               = 5000        ; Save energies every 10 ps
# nstlog                  = 5000        ; Save log every 10 ps
# nstxout-compressed      = 5000        ; Save compressed trajectory every 10 ps
# compressed-x-precision  = 1000
# continuation            = yes         ; Restarting simulation
# constraint_algorithm    = lincs
# constraints             = h-bonds     ; Only constrain bonds with hydrogens
# lincs_iter              = 1
# lincs_order             = 4
# cutoff-scheme           = Verlet
# nstlist                 = 20
# ns_type                 = grid
# rlist                   = 1.0
# rvdw                    = 1.0
# coulombtype             = PME
# rcoulomb                = 1.0
# vdw-modifier            = Force-switch
# rvdw-switch             = 0.9
# tcoupl                  = V-rescale
# tc-grps                 = Protein Non-Protein
# tau-t                   = 0.1 0.1
# ref-t                   = 300 300
# pcoupl                  = Parrinello-Rahman
# pcoupltype              = isotropic
# tau-p                   = 2.0
# ref-p                   = 1.0
# compressibility         = 4.5e-5
# refcoord_scaling        = com
# gen_vel                 = no
#

!gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 1

!gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 12

!gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 12 -cpi md.cpt -noappend

!gmx trjcat -f md.xtc md.part0002.xtc md.part0003.xtc md.part0004.xtc md.part0005.xtc md.part0006.xtc md.part0007.xtc -o full_md.xtc -cat

`

!echo -e "Protein\nProtein" | gmx rms -s md.tpr -f full_md.xtc -o rmsd_pocket.xvg -tu ns

!echo 4 1 | gmx trjconv -s md.tpr -f full_md.xtc -o aligned.xtc -fit rot+trans

!echo 1 | gmx rmsf -s md.tpr -f aligned.xtc -o rmsf_protein.xvg -res

!echo "Protein" | gmx sasa -s md.tpr -f full_md.xtc -o sasa_pocket.xvg -or sasa_residue.xvg -surface 'Protein'

import matplotlib.pyplot as plt

def read_xvg(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [line.strip().split() for line in lines if not line.startswith(('#', '@'))]
    x = [float(i[0]) for i in data]
    y = [float(i[1]) for i in data]
    return x, y

# RMSD Plot
rmsd_x, rmsd_y = read_xvg('/content/drive/MyDrive/Rayca_eval/MD_simulation/rmsd_pocket.xvg')
plt.figure(figsize=(8, 5))
plt.plot(rmsd_x, rmsd_y)
plt.xlabel('Time (ns)')
plt.ylabel('RMSD (nm)')
plt.title('Pocket RMSD')
plt.grid(True)
plt.savefig('/content/drive/MyDrive/Rayca_eval/analysis_plots/rmsd_pocket.png')
plt.show()
plt.close()

# RMSF Plot
rmsf_x, rmsf_y = read_xvg('/content/drive/MyDrive/Rayca_eval/MD_simulation/rmsf_protein.xvg')
plt.figure(figsize=(8, 5))
plt.plot(rmsf_x, rmsf_y)
plt.xlabel('Residue Index')
plt.ylabel('RMSF (nm)')
plt.title('Pocket RMSF')
plt.grid(True)
plt.savefig('/content/drive/MyDrive/Rayca_eval/analysis_plots/rmsf_pocket.png')
plt.show()
plt.close()

# SASA Plot
sasa_x, sasa_y = read_xvg('/content/drive/MyDrive/Rayca_eval/MD_simulation/sasa_pocket.xvg')
plt.figure(figsize=(8, 5))
plt.plot(sasa_x, sasa_y)
plt.xlabel('Time (ns)')
plt.ylabel('SASA (nm²)')
plt.title('Pocket SASA')
plt.grid(True)
plt.savefig('/content/drive/MyDrive/Rayca_eval/analysis_plots/sasa_pocket.png')
plt.show()
plt.close()

#continuous Residue number plot RMSF

import numpy as np
import matplotlib.pyplot as plt

# Load RMSF data (residue numbers and values)
data = np.loadtxt('/content/drive/MyDrive/Rayca_eval/MD_simulation/rmsf_pocket.xvg', comments=['#', '@'])

residue_numbers = data[:, 0]  # Original residue numbers
rmsf_values = data[:, 1]      # RMSF values

# Create a continuous x-axis (1,2,3,...)
x_axis = np.arange(1, len(residue_numbers) + 1)  # Continuous numbering

# Plot RMSF with continuous indexing
plt.figure(figsize=(8,5))
plt.plot(x_axis, rmsf_values, marker='o', linestyle='-', color='b', label="RMSF")
plt.xlabel("Residue Index (Pocket)")
plt.ylabel("RMSF (nm)")
plt.title("Pocket RMSF (Fixed x-axis)")
plt.grid(True)
plt.legend()

# Save the fixed plot
plt.savefig("/content/drive/MyDrive/Rayca_eval/analysis_plots/rmsf_pocket_fixed.png")
plt.show()

from MDAnalysis.analysis import rms, contacts, hole2, align, pca, distances

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

# Define file paths
tpr_file = "/content/drive/MyDrive/Rayca_eval/MD_simulation/md.tpr"
xtc_file = "/content/drive/MyDrive/Rayca_eval/MD_simulation/full_md.xtc"

# Load trajectory
u = mda.Universe(tpr_file, xtc_file)

# Define pocket atoms (assuming it's already selected in the simulation)
pocket = u.select_atoms("protein and around 5.0 resid 25-216")  # Adjust selection if needed

print(f"Loaded {len(pocket)} pocket atoms from trajectory.")

# Align trajectory to first frame
R = rms.RMSD(u, u, select="protein and around 5.0 resid 25-216", ref_frame=0)
R.run()

# Extract RMSD data
rmsd_time = R.rmsd[:, 0] / 1000  # Convert ps to ns
rmsd_values = R.rmsd[:, 2]  # RMSD (nm)

# Plot RMSD
plt.figure(figsize=(8, 5))
plt.plot(rmsd_time, rmsd_values, label="Pocket RMSD", color='#377eb8', linewidth=2.5)
plt.xlabel("Time (ns)", fontsize=14)
plt.ylabel("RMSD (nm)", fontsize=14)
plt.title("Pocket RMSD Over Time", fontsize=16)
plt.legend(frameon=False, fontsize=12)
plt.savefig("/content/drive/MyDrive/Rayca_eval/analysis_plots/rmsd_pocket_traj.png", dpi=300, bbox_inches='tight')
plt.show()

from MDAnalysis.analysis.rms import RMSF

# Calculate RMSF
rmsf = RMSF(pocket).run()

# Plot RMSF
plt.figure(figsize=(8, 5))
plt.plot(pocket.resids, rmsf.rmsf, label="Pocket RMSF", color='#e41a1c', linewidth=2.5)
plt.xlabel("Residue Index (Pocket)", fontsize=14)
plt.ylabel("RMSF (nm)", fontsize=14)
plt.title("Pocket Residue Fluctuations (RMSF)", fontsize=16)
plt.legend(frameon=False, fontsize=12)
#plt.savefig("/content/drive/MyDrive/Rayca_eval/analysis_plots/rmsf_pocket_traj.png", dpi=300, bbox_inches='tight')
plt.show()

import MDAnalysis as mda
import freesasa
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os

# Load Universe
tpr_file = "/content/drive/MyDrive/Rayca_eval/MD_simulation/md.tpr"
xtc_file = "/content/drive/MyDrive/Rayca_eval/MD_simulation/full_md.xtc"
u = mda.Universe(tpr_file, xtc_file)

# Define your pocket selection (use appropriate atom/group selection)
pocket = u.select_atoms("protein")

# Arrays to store results
sasa_values = []
times = []

# Create temporary directory for storing temp PDBs
with tempfile.TemporaryDirectory() as tmpdir:
    for ts in u.trajectory:
        pdb_path = os.path.join(tmpdir, "frame.pdb")

        # Write current pocket frame to temp PDB file
        pocket.write(pdb_path)

        # Load into FreeSASA
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)

        sasa_values.append(result.totalArea())
        times.append(ts.time / 1000)  # Convert ps to ns

# Plot SASA
plt.figure(figsize=(8, 5))
plt.plot(times, sasa_values, label="Pocket SASA", color='#66c2a5', linewidth=2.5)
plt.xlabel("Time (ns)", fontsize=14)
plt.ylabel("SASA (nm²)", fontsize=14)
plt.title("Pocket SASA Over Time", fontsize=16)
plt.legend(frameon=False, fontsize=12)
plt.savefig("/content/drive/MyDrive/Rayca_eval/analysis_plots/sasa_pocket_traj.png", dpi=300, bbox_inches='tight')
plt.show()

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  

# Load your trajectory
u = mda.Universe("md.tpr", "full_md.xtc")
pocket = u.select_atoms("protein")  # Adjust if needed

# Collect pocket atom positions over time
coords = []

for ts in u.trajectory[::10]:  # stride to speed up
    coords.append(pocket.positions.copy())

coords = np.array(coords)  # shape: (n_frames, n_atoms, 3)
n_frames = coords.shape[0]

# Flatten for PCA
coords_2d = coords.reshape(n_frames, -1)

# Run PCA
pca_model = PCA(n_components=2)
pca_projection = pca_model.fit_transform(coords_2d)

# Plot PCA projection
plt.figure(figsize=(8, 5))
sc = plt.scatter(
    pca_projection[:, 0], pca_projection[:, 1],
    c=np.linspace(0, n_frames / 10, n_frames),  # time in ns
    cmap='viridis', alpha=0.85
)
plt.xlabel("PC1", fontsize=14)
plt.ylabel("PC2", fontsize=14)
plt.title("PCA: Pocket Shape Evolution", fontsize=16)
plt.colorbar(sc, label="Time (ns)")
plt.savefig("/content/drive/MyDrive/Rayca_eval/analysis_plots/pocket_pca_traj.png", dpi=300, bbox_inches="tight")
plt.show()

import MDAnalysis as mda
import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis.analysis import distances

# Load your simulation
u = mda.Universe("md.tpr", "full_md.xtc")
pocket = u.select_atoms("protein")  # adjust selection if needed

volumes = []
time_ns = []

for ts in u.trajectory[::10]:  # stride to speed things up
    dist_matrix = distances.distance_array(pocket.positions, pocket.positions)
    min_dist = dist_matrix[np.nonzero(dist_matrix)].min()  # avoid 0s from self-distance
    volumes.append(min_dist ** 3)  # cube of smallest distance as approx. volume
    time_ns.append(ts.time / 1000)  # convert to ns

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(time_ns, volumes, color='#984ea3', linewidth=2.5)
plt.xlabel("Time (ns)", fontsize=14)
plt.ylabel("Approximate Volume (nm³)", fontsize=14)
plt.title("Pocket Volume Over Time", fontsize=16)
plt.savefig("/content/drive/MyDrive/Rayca_eval/analysis_plots/pocket_volume_traj.png", dpi=300, bbox_inches="tight")
plt.show()
