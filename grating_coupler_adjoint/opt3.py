import meep as mp
import meep.adjoint as mpa
import numpy as np
from autograd import numpy as npa
from autograd import tensor_jacobian_product
import nlopt
import os
import json
import matplotlib.pyplot as plt
import warnings
import gc

import params as p  

def main():
    os.makedirs("png", exist_ok=True)
    warnings.filterwarnings("ignore")
   
    design_region_width_x = 6.0
    design_region_width_z = 2.0
    design_region_resolution = 10

    Nx = int(design_region_width_x * design_region_resolution) + 1
    Nz = int(design_region_width_z * design_region_resolution) + 1
    n = Nx * Nz

    design_variables = mp.MaterialGrid(
        mp.Vector3(Nx, 0, Nz),
        p.Air,
        p.Si,
        grid_type="U_MEAN",
    )

    wg_w = 2.0  

    design_region = mpa.DesignRegion(
        design_variables,
        volume=mp.Volume(
            center=mp.Vector3(1.0, 0, 0),
            size=mp.Vector3(design_region_width_x, p.h_si, design_region_width_z),
        ),
    )


    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, p.h_box, mp.inf),
            center=mp.Vector3(0, -p.h_si / 2 - p.h_box / 2, 0),
            material=p.SiO2,
        ),
        mp.Block(
            size=mp.Vector3(10, p.h_si, wg_w),  
            center=mp.Vector3(0, 0, 0),
            material=p.Si,
        ),
        mp.Block(
            center=design_region.center,
            size=design_region.size,
            material=design_variables,
        ),
    ]

    fiber_core = mp.Cylinder(
        radius=1,
        height=5,
        axis=mp.Vector3(np.sin(p.theta), np.cos(p.theta), 0),
        center=mp.Vector3(1.5, 3, 0),  
        material=mp.Medium(index=p.fiber_core_n),
    )
    geometry.append(fiber_core)

    fcen = 1 / 1.55
    kdir = mp.Vector3(np.sin(p.theta), -np.cos(p.theta), 0)

    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=0.2),
            center=mp.Vector3(1.5, 3, 0),
            size=mp.Vector3(3.0, 0, 3.0),
            direction=mp.Y,
            eig_kpoint=kdir,
            eig_band=1,
            eig_match_freq=True,
        )
    ]

    pml_thickness = 1.0
    cell_sx, cell_sy, cell_sz = 12.0, 14.0, 6.0

    sim = mp.Simulation(
        sources=sources,
        cell_size=mp.Vector3(cell_sx, cell_sy, cell_sz),
        geometry=geometry,
        resolution=design_region_resolution,
        boundary_layers=[mp.PML(pml_thickness)],
        eps_averaging=True,
    )

    monitor_x = -3

    out_monitor = mpa.EigenmodeCoefficient(
        sim,
        mp.Volume(
            center=mp.Vector3(monitor_x, 0, 0),
            size=mp.Vector3(0, p.h_si * 4, wg_w + 1.0),
        ),
        mode=1,
        forward = False
    )

    in_monitor = mpa.EigenmodeCoefficient(
    sim,
    mp.Volume(
        center=mp.Vector3(1.5, 1.8, 0),   
        size=mp.Vector3(2.0, 0, 2.0),
    ),
    mode=1,
    forward=True
    )
    
    cur_itr = [0]


    def J(out):
        coupled_mode = out[0]
        power = npa.abs(coupled_mode) ** 2
        print("out: ", out)

        return ( power )


    filter_radius = 0.05

    def mapping(x, eta, beta):
        filtered = mpa.conic_filter(
            x, filter_radius,design_region_width_x, design_region_width_z, design_region_resolution
        )
        projected = mpa.tanh_projection(filtered, beta, eta)
        return projected.flatten()

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=[out_monitor],
        design_regions=[design_region],
        frequencies=[fcen],
        decay_by=1e-4,
    )

    history = []

    def f(v, gradient, cur_beta):
        print(f"Iteration: {cur_itr[0] + 1} | Beta: {cur_beta}")
        
        plt.figure()
        opt.plot2D(
            output_plane=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(cell_sx, 0, cell_sz),
            )
        )
        plt.savefig(f"png/iter_1{cur_itr[0]:03d}.png", dpi=150)
        plt.close()
        
        plt.figure()
        opt.plot2D(
            output_plane=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(cell_sx, cell_sy, 0),
            )
        )
        plt.savefig(f"png/iter_2{cur_itr[0]:03d}.png", dpi=150)
        plt.close()
        

        f0, dJ_du = opt([mapping(v, 0.5, cur_beta)])

        
        history.append({"iteration": int(cur_itr[0]), "fitness": float(f0)})
        with open("history.json", "w") as fj:
            json.dump(history, fj, indent=2)

        
        if gradient.size > 0:
            g = tensor_jacobian_product(mapping, 0)(v, 0.5, cur_beta, dJ_du)
            gradient[:] = np.nan_to_num(np.array(g), nan=0.0, posinf=0.0, neginf=0.0)

        sim.filename_prefix = f"png/iter_{cur_itr[0]:04d}"
        mp.output_epsilon(sim)
                
        plt.figure()
        opt.plot2D(
            output_plane=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(cell_sx, 0, cell_sz),
            )
        )
        plt.savefig(f"png/iter_{cur_itr[0]:03d}.png", dpi=150)
        plt.close()
        
        plt.figure()
        opt.plot2D(
            output_plane=mp.Volume(
                center=mp.Vector3(0, 0, 0),
                size=mp.Vector3(cell_sx, cell_sy, 0),
            )
        )
        plt.savefig(f"png/iter_x{cur_itr[0]:03d}.png", dpi=150)
        plt.close()
        
        gc.collect()
        
        # if cur_itr[0] % 2 == 0:
        #     opt.sim.reset_meep()

        #     mapped = mapping(v, .5, cur_beta)

        #     opt.update_design([mapped])

        #     opt.forward_run()   

        #     opt.sim.filename_prefix = f"png/iter_{cur_itr[0]:04d}"
        #     sim.reset_meep()

        #     ez_plane = mp.Volume(
        #          center=mp.Vector3(0, 0, 0),          # y = 0 plane
        #             size=mp.Vector3(cell_sx, cell_sy, 0) # 0 thickness in y → x–z plane
        #         )

#             opt.sim.run(
#             mp.to_appended(
#              f"ez-{cur_itr[0]:04d}",
#                 mp.at_every(
#              5,
#                  mp.in_volume(ez_plane, mp.output_efield_z)
#                     )
#                     ),
#                     until=100
# )

        cur_itr[0] += 1
        return float(f0)

   
    algorithm = nlopt.LD_MMA

    x = np.ones(n) * 0.5
    
    # approx_pitch = 0.5 
    # x_grid = np.linspace(0, design_region_width_x, Nx)
    
    # grating_1d = 0.5 + 0.3 * np.sin(2 * np.pi * x_grid / approx_pitch)
    
    # x_2d = np.tile(grating_1d[:, np.newaxis], (1, Nz))
    
    # x = x_2d.flatten()
    
    
    lb = np.zeros(n)
    ub = np.ones(n)

    betas = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    evals_per_beta = 30  
    
    for cur_beta in betas:
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)
        solver.set_vector_storage(20)

        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(evals_per_beta)
    
        x[:] = solver.optimize(x)
            

    np.save("final_design.npy", x)
    print("done")


if __name__ == "__main__":
    main()