import meep as mp
import numpy as np
import os

resolution = 20
w = 1
l = 20
dpml = 1
design_region_dimension = 5

sx = l + design_region_dimension
sy = 16

cell_size = mp.Vector3(sx, sy, 0)
pml_layers = [mp.PML(dpml)]

fcen = 1 / 1.55

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution,
    dimensions=2,
    epsilon_input_file="./log/3/png/iter_0276-eps-000623.35.h5",   # <-- full-cell epsilon
)

# ===============================
# SOURCE (same as optimization)
# ===============================
sources = [
    mp.EigenModeSource(
        mp.GaussianSource(fcen, fwidth=0.2 * fcen),
        center=mp.Vector3(-10, -2),
        size=mp.Vector3(y=w * 3),
        direction=mp.X,
        eig_band=1,
    )
]

sim.sources = sources

# ===============================
# MODE MONITORS
# ===============================
input_mode = sim.add_mode_monitor(
    fcen, 0, 1,
    mp.ModeRegion(
        center=mp.Vector3(-8, -2),
        size=mp.Vector3(0, w * 2),
    )
)

output_mode = sim.add_mode_monitor(
    fcen, 0, 1,
    mp.ModeRegion(
        center=mp.Vector3(10, 2),
        size=mp.Vector3(0, w * 2),
    )
)

# ===============================
# OPTIONAL: SAVE EPS + EZ SNAPSHOTS
# ===============================
output_dir = "forward_validation"
os.makedirs(output_dir, exist_ok=True)
sim.use_output_directory(output_dir)

# ===============================
# RUN SIMULATION
# ===============================
# sim.run(
#     mp.at_beginning(mp.output_epsilon),
#     mp.at_every(5, mp.output_efield_z),
#     until_after_sources=200
# )
sim.run(
    mp.at_beginning(mp.output_epsilon),
    until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ez, mp.Vector3(10,2), 1e-6
    )
)

# ===============================
# COMPUTE TRANSMISSION
# ===============================
coeff_in = sim.get_eigenmode_coefficients(input_mode, [1])
coeff_out = sim.get_eigenmode_coefficients(output_mode, [1])

print ("coeff_in", coeff_in)
Pin = np.abs(coeff_in.alpha[0, 0, 0])**2
Pout = np.abs(coeff_out.alpha[0, 0, 0])**2

print("\n========== RESULTS ==========")
print("Input power  =", Pin)
print("Output power =", Pout)
print("Transmission =", Pout / Pin)
print("=============================\n")
print("backward:", np.abs(alpha[0,0,1])**2)
