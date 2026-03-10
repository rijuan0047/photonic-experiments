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

SiO2 = mp.Medium(index=1.4)
Si = mp.Medium(index=3.4)
Air = mp.Medium(index = 1)

def main( ) :
    os.makedirs("png", exist_ok=True)
    warnings.filterwarnings("ignore") 
    
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

    design_variables = mp.MaterialGrid(mp.Vector3(Nx, Ny), Air, Si, grid_type = "U_MEAN")
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
        center=deisgn_region.center, size=deisgn_region.size, material=design_variables
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

    input_monitor_x = -8
    output_monitor_x = 10

    TE_source = mpa.EigenmodeCoefficient(
        sim,
        mp.Volume(center=mp.Vector3(input_monitor_x, -2, 0), size=mp.Vector3(y=w * 2)),
        mode=1
    )

    TE0 = mpa.EigenmodeCoefficient(
        sim,
        mp.Volume(center=mp.Vector3(output_monitor_x, 2, 0), size=mp.Vector3(y=( w +1  )* 2)),
        mode=3
    )


    obj_list = [TE_source, TE0]
    
    current_source_eta = 0
    current_out_waveguide_eta = 0
    
    is_forward_call = True
    
    alpha = [1e-3]   

    def J(source, out):
        eps = 1e-9
        nonlocal current_source_eta, current_out_waveguide_eta, is_forward_call

        source = source[0]   # source returns array of one element
        out = out[0]

         
        if is_forward_call == True:
            current_source_eta = np.abs(source) ** 2
            current_out_waveguide_eta = np.abs(out) ** 2
            
            # is_forward_call = False

        return npa.abs(out)**2 / 67

     
    eta_e = 0.55
    eta_i = 0.5
    eta_d = 1 - eta_e
    minimum_length = 1

    filter_radius = mpa.get_conic_radius_from_eta_e(minimum_length, eta_e)
    
    pixel_size = 1.0 / design_region_resolution
    filter_radius = 5 * pixel_size

    def mapping(x, eta, beta):

    # filter

        filtered_field = mpa.conic_filter(
        x, filter_radius, design_region_dimension, design_region_dimension + 4, design_region_resolution
    )

    # projection 

        projection_field = mpa.tanh_projection(filtered_field, beta, eta)

        return projection_field.flatten()

    opt = mpa.OptimizationProblem(
        simulation=sim,
        objective_functions=J,
        objective_arguments=obj_list,
        design_regions=[deisgn_region],
        frequencies=[fcen],
        decay_by=1e-5,
    )
    
    plt.figure()
    sim.plot2D()
    plt.show()
    
    np.random.seed(1)
    x0 = np.random.rand(Nx * Ny)
    x0 =x0.flatten()
    
    opt.update_design([x0])
    opt.plot2D(plot_sources=False)
    plt.show()
    
    cur_itr = [0]

    eff_history = []
    history = []

    def f(v, gradient, cur_beta):
        print(f"current iter : {cur_itr[0] + 1}")

        f0, dJ_du = opt([mapping(v, eta_i, cur_beta)]) 

        plt.figure()
        ax = plt.gca()
        opt.plot2D(
        False,
        ax=ax,
        plot_sources_flag=False,
        plot_monitors_flag=False,
        plot_boundaries_flag=False,
        )

        plt.savefig(f"png/iter_{cur_itr[0]:04d}.png")
        
        source_coeff = TE_source()
        out_coeff = TE0()
        
        source_power = float(np.abs(source_coeff[0])**2)
        out_power = float(np.abs(out_coeff[0])**2)

        grad_np = np.array(dJ_du, dtype=float)
        grad_np_abs = np.sum(np.abs(grad_np)) 
        
        np.save(f"png/gradient_{cur_itr[0]:04d}.npy", grad_np.reshape(Nx, Ny))
        np.save(f"png/x_{cur_itr[0]:04d}.npy", np.rot90(v.reshape(Nx, Ny)))

                
        sim.filename_prefix = f"png/iter_{cur_itr[0]:04d}"
        mp.output_epsilon(sim)
        eff = float(np.real(f0))
        
        if cur_itr[0] % 2 == 0:
            opt.sim.reset_meep()

            mapped = mapping(v, eta_i, cur_beta)

            opt.update_design([mapped])

            opt.forward_run()   

            opt.sim.filename_prefix = f"png/iter_{cur_itr[0]:04d}"
            sim.reset_meep()

            opt.sim.run(
              mp.to_appended(
            f"ez-{cur_itr[0]:04d}",
            mp.at_every(0.6, mp.output_efield_z)
                ),
                until=200
            )
            
        design = mapping(v, eta_i, cur_beta).reshape(Nx, Ny)

        si_strong = design > 0.9
        air_strong = design < 0.1

        si_ratio = float(np.mean(si_strong))
        air_ratio = float(np.mean(air_strong))
        gray_ratio = 1.0 - si_ratio - air_ratio
        binarity = si_ratio + air_ratio

        print("eta: from f", current_source_eta, current_out_waveguide_eta )
        entry = {
        "iteration": int(cur_itr[0] + 1),
        "beta": float(cur_beta),
        "fitness": float(eff),
        "binarity": float(binarity),
        "current_source_eta": str(current_source_eta),
        "current_out_waveguide_eta": str(current_out_waveguide_eta),
        "grad_L1_norm": grad_np_abs,
        "relative_grad_L1_norm": grad_np_abs / (eff + 1e-9)
        
    }

        history.append(entry)

        with open ("history.json", "w") as f:
          json.dump(history, f, indent = 2)

        if gradient.size > 0:
             g = gradient[:] = tensor_jacobian_product(mapping, 0)(
            v, eta_i, cur_beta, dJ_du
        )

        gradient[:] = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        cur_itr[0] = cur_itr[0] + 1
    
        png_name = f"png/iter_{cur_itr[0]:04d}.png"
        plt.savefig(png_name, dpi=200, bbox_inches="tight") # bounding box inches tight: it crops the extra whitespace 
        plt.close()
        fig = plt.figure()

        design = mapping(v, eta_i, cur_beta).reshape(Nx, Ny)

        gc.collect()

        design = mapping(v, eta_i, cur_beta).reshape(Nx, Ny)

        return float(np.real(f0))
    
    
    # algorithm = nlopt.LD_MMA
    algorithm = nlopt.LD_LBFGS

    

    n = Nx * Ny 

    x = np.ones(n) * .5
    # x = np.random.rand(n)

    # x = np.random.rand(n)*0.2 + 0.4
    
    lb = np.zeros(n)
    ub = np.ones(n)

    cur_beta = 2
    beta_scale = 3
    num_betas = 10
    update_factor = 50


    for iters in range(num_betas):
        solver = nlopt.opt(algorithm, n)
        solver.set_lower_bounds(lb)
        solver.set_upper_bounds(ub)

        solver.set_vector_storage(20)  

        solver.set_max_objective(lambda a, g: f(a, g, cur_beta))
        solver.set_maxeval(update_factor)
        
        solver.set_ftol_rel(1e-6)
        solver.set_xtol_rel(1e-6)

   

        # x[:] = solver.optimize(x)
        
        try:
            x[:] = solver.optimize(x)
        except nlopt.runtime_error:
            print("NLopt finished (normal termination)")

        cur_beta *= beta_scale

    
if __name__ == "__main__" : 
    main()