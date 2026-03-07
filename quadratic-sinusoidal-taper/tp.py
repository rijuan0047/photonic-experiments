import numpy as np
import matplotlib.pyplot as plt
import meep as mp

def waveguide_to_source_power_ratio(a= 0 , b = 0 , c = 7):
    
    """
    Calculates the transmission efficiency of a tapered waveguide.
    'a' , 'b', and 'c', are parameters controlling the taper's geometry
    """
    c = c * 20
    
    design_resolution = 1000
    mp.verbosity(0) # turns off meep's terminal outputs for cleaner terminal

    # longitudinal/ propagation direction axis 
    actual_z = np.linspace(0, 10 ,design_resolution)
    normalized_z = actual_z / 10
    actual_z = actual_z - 5 # taper geometry starts from -5

    # X is normalized width profile (along the propagation direction)
    X = a * (b * normalized_z ** 2 + (1 - b )  * normalized_z) + (1 - a) * np.sin (c * np.pi * .5 * normalized_z )  ** 2
    actual_width_profile = 10  + (.5 - 10) * X
    half_widths = actual_width_profile / 2

    # visulazing the actual width profile along the propagation direction 
    # plt.plot(actual_width_profile  )
    # plt.savefig("x.png")

    # construction of geometry 
    # tapered section
    vertices = [mp.Vector3(z, y) for z, y in zip(actual_z, half_widths )]
    vertices += [mp.Vector3(z, -y) for z, y in zip(reversed(actual_z), reversed(half_widths))]

    taper_polygon = mp.Prism(
    vertices=vertices,
    material=mp.Medium(index = 3.47)   ,
    height = mp.inf   
    )

    # 10 micron waveguide section (input side)
    larger_box = mp.Prism(
    vertices = [mp.Vector3(-15, -5) , mp.Vector3(-5, -5) , mp.Vector3(-5, 5), mp.Vector3(-15, 5)],
    material=mp.Medium(index = 3.47)   ,
    height = mp.inf   
    )

    # 500 nm wavegude section (output side)
    smaller_box = mp.Prism(
    vertices = [mp.Vector3(5 , .25) , mp.Vector3(15, .25), mp.Vector3(15, -.25) , mp.Vector3(5, -.25)],
    material=mp.Medium(index = 3.47)   ,
    height = mp.inf  
    )

    # geometry array to be passed to simulation object constructor 
    geometry = [larger_box, taper_polygon, smaller_box]

    # fdtd simulation at a single wavelength 
    wavelength = 1.55
    fcen = 1.0 / wavelength
    df = 0.1 * fcen  

    # eigen mode source for fdtd 
    sources = [
     mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(-9, 0),
        size=mp.Vector3(0, 13),
        direction=mp.X,
        eig_band=1 
    )
    ]

    cell_size = mp.Vector3(10 + 16, 16, 0)

    # setting up the simulation object 
    sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=[mp.PML(2.0)],
    geometry=geometry,
    sources=sources,
    resolution=200               
    )

    # monitor for the output waveguide mode 1 power measurement
    monitor_center = mp.Vector3(7, 0)
    monitor_size = mp.Vector3(0, 2)

    flux_monitor = sim.add_mode_monitor(
    fcen, 0, 1,
    mp.ModeRegion(center=monitor_center, size=monitor_size)
    )

    # monitor for the source waveguide mode 1 power measurement
    source_monitor_center = mp.Vector3(-7, 0)
    source_monitor_size = mp.Vector3(0, 15)

    source_flux_monitor = sim.add_mode_monitor(
    fcen, 0, 1,
    mp.ModeRegion(center=source_monitor_center, size=source_monitor_size)
    )

    # visualizing the geometry 
    plt.figure()
    sim.plot2D()
    plt.savefig("optimized_geometry.png")

    stop_condition = mp.stop_when_fields_decayed(5000, mp.Ez, monitor_center, 1e-6)

    # sim.run(
    #     # mp.to_appended("ez_field", mp.at_every(5, mp.output_efield_z)),
    #     until_after_sources=stop_condition  
    # )
    
    # res = sim.get_eigenmode_coefficients(flux_monitor, [1])
    # raw_transmission_power = abs(res.alpha[0,0,0])**2

    # res_source = sim.get_eigenmode_coefficients(source_flux_monitor, [1])
    # source_transmission = abs(res_source.alpha[0,0,0])**2
    # print("source: ", source_transmission )

    # print("waveguide: ", raw_transmission_power)
  
    # return float (raw_transmission_power / source_transmission )
    

if __name__ == "__main__":
    # waveguide_to_source_power_ratio( 0.3572377911216459,
    #                 0.7655866019796231,
    #                 5)
     waveguide_to_source_power_ratio( 0.47,
                    0.51,
                    .76)