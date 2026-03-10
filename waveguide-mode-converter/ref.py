import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import meep.adjoint as mpa


resolution = 15

w = 2

dpml = 1

l = 20

sy = 8
sx = l + 2 * dpml

cell_size = mp.Vector3(sx, sy, 0)

pml_layers = [mp.PML(dpml)]

fcen = 1 / 1.55

SiO2 = mp.Medium(index=1.4)
Si = mp.Medium(index=3.4)

sources = [
    mp.EigenModeSource(
        mp.GaussianSource(fcen, fwidth=0.2 * fcen),
        center=mp.Vector3(x = -5, y = 0),
        size=mp.Vector3( y = w + 2 ),
        eig_band = 2
    )
]

geometry = [
    mp.Block(
        size=mp.Vector3( 2 * sx , w, 0),
        center=mp.Vector3( x = - 8),
        material=Si,
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

sim.use_output_directory("refrence_run")

monitor = sim.add_mode_monitor(fcen, 0, 1, mp.ModeRegion(center = mp.Vector3(5, 0, 0), size = mp.Vector3(0, w + 2) ), )

monitor_mpa = TE0 = mpa.EigenmodeCoefficient(
        sim,
        mp.Volume(center = mp.Vector3(5, 0, 0), size = mp.Vector3(0, w + 2)),
        mode=2,
        frequencies=[fcen]
    )
    
sim.run(mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)), until_after_sources=mp.stop_when_fields_decayed(
        50,         
        mp.Ez,       
        mp.Vector3(5, 0, 0),  
        1e-6        
    ))
plt.figure()
sim.plot2D(fields = mp.Ez)
plt.show()

res = sim.get_eigenmode_coefficients(
    monitor, [2]
)

print (monitor_mpa())


coeff = res.alpha
input_flux = np.abs(coeff[0, :, 0]) ** 2

print(coeff)
print(input_flux)