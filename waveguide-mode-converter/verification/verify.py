import meep as mp
import meep.adjoint as mpa
import numpy as np
import matplotlib.pyplot as plt

resolution = 20

w = 1
l = 20

data = np.load("./../png/projected_design_0351.npy")
print(data.shape, data.min(), data.max())
dpml = 1
design_region_dimension = 5

sx = l + design_region_dimension
sy = 16

cell_size = mp.Vector3(sx, sy, 0)
pml_layers = [mp.PML(dpml)]
fcen = 1 / 1.55

SiO2 = mp.Medium(index=1.4)
Si = mp.Medium(index=3.4)
Air = mp.Medium(index = 1)

kpoint = mp.Vector3(1, 0, 0)
    
sources = [
    mp.EigenModeSource(
        mp.GaussianSource(fcen, fwidth=0.2 * fcen),
        center=mp.Vector3(x = -10, y = -2),
        size=mp.Vector3(y= w * 3),
        eig_kpoint=kpoint,
        eig_band=1,
        direction=mp.X,
    )
    ]
    
design_region_resolution = resolution

Nx = int( design_region_dimension * design_region_resolution) + 1
Ny = int( (design_region_dimension + 4 ) * design_region_resolution) + 1

design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), Air, Si, grid_type = "U_MEAN", weights = data)
deisgn_region = mpa.DesignRegion(
    design_variables,
    volume=mp.Volume(center=mp.Vector3(), size=mp.Vector3(design_region_dimension, design_region_dimension + 4, 0)),
    )
    
overlap = 0.5
    
geometry = [
        mp.Block(
        size=mp.Vector3(l/2 + overlap + 1  , w, 0),
        center=mp.Vector3(x = -design_region_dimension/2 -l/4 + overlap / 2 + 0.5 , y = -2),
        material=Si,
    ),
       
        mp.Block(
        size=mp.Vector3(l/2 + overlap, w + 1, 0),
        center=mp.Vector3(x = design_region_dimension/2 + l/4 - overlap / 2, y = 2),
        material=Si,
    ),
        
        mp.Block(
        center=mp.Vector3(), material=design_variables,
        size=mp.Vector3(design_region_dimension, design_region_dimension + 4, 0)
    ),
    ]
    
sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        resolution=resolution,
        dimensions=2,
        geometry=geometry,
    )

sim.plot2D(cmap = "viridis" )
plt.savefig("design-verification.png", dpi = 400)

# trans_fr = mp.FluxRegion(center=mp.Vector3(10, 2, 0), size=mp.Vector3(y=(w + 1) * 2))
trans = sim.add_mode_monitor(
    fcen,
    0,
    1,
    mp.ModeRegion(
        center=mp.Vector3(10, 2, 0),
        size=mp.Vector3(0, (w+1)*2, 0)
    )
)
# trans = sim.add_mode_monitor(fcen, .2, 1, trans_fr)

source = sim.add_mode_monitor(
    fcen,
    0,
    1,
    mp.ModeRegion(
       center=mp.Vector3(-8, -2, 0), size=mp.Vector3(y=w * 2)
    )
)


sim.run(
        mp.to_appended(
        f"ez",
            mp.at_every(0.6, mp.output_efield_z)
                ),
                until=500
            )


res = sim.get_eigenmode_coefficients(trans,[1])
res_s = sim.get_eigenmode_coefficients(source,[1])

print(res.alpha[:, :, 0])

alpha = res.alpha[0,0,:]
alpha_source = res_s.alpha[0,0,:]

forward = np.abs(alpha[0]) ** 2
backward = np.abs(alpha[1]) ** 2

print("Forward mode amplitude:", forward)
print("Backward mode amplitude:", backward)

forward = np.abs(alpha_source[0]) ** 2
backward = np.abs(alpha_source[1]) ** 2

print("source Forward mode amplitude:", forward)
print("source Backward mode amplitude:", backward)

