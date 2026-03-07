import numpy as np
from cmaes import CMA
import json
from multiprocessing import Pool

import tp

pop_size = 10

def objective(x) :
    val = tp.waveguide_to_source_power_ratio(x[0] , x[1], x[2])
    return -val

optimizer = CMA(mean = np.ones(3) * .5 , sigma = .5, population_size = pop_size,  bounds=np.array([[0,1],[0,1] , [0, 1]]))

history = []

for generation in range(100):
    
    json_data = []
    
    individuals = [optimizer.ask() for _ in range(pop_size) ]
    
    with Pool(pop_size) as pool: 
        values = pool.map(objective , individuals)
        
    solutions = list(zip(individuals, values))
    
    optimizer.tell(solutions)
    
    for (x , val) in solutions:
        json_data.append({
            "params": x.tolist(),
            "score": val
            
        })
    
    entry = {
        "generation": f"{generation}",
        "solutions": json_data
    }
    
    history.append(entry)

    with open("history.json" , "w") as f:
        json.dump(history , f, indent= 4)
