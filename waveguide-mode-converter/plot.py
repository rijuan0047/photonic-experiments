import json
import matplotlib.pyplot as plt
import numpy as np

beta_list = []
fitness_list = []
iteration_list = []
binarity_list = []
normalized_grad_norm_L1 = []
grad_norm_L1 = []

with open("./history.json", "r") as historyFile:
    data = json.load(historyFile)

    for iteration_data in data:
        beta_list.append(iteration_data["beta"])
        fitness_list.append(iteration_data["fitness"])
        iteration_list.append(iteration_data["iteration"])
        binarity_list.append(iteration_data["binarity"])
        normalized_grad_norm_L1.append(iteration_data["relative_grad_L1_norm"])
        grad_norm_L1.append(iteration_data["grad_L1_norm"])

max_beta = max(beta_list)
grad_max = max(normalized_grad_norm_L1)
beta_list_normalized = [b /  max_beta for b in beta_list]
grad_norm_normalized = [b /  grad_max for b in normalized_grad_norm_L1]

start = 200
end = 500

x = iteration_list[start : end]

plt.figure()
plt.plot(x ,  beta_list_normalized[start:end], label = "beta")
plt.plot(x , fitness_list[start:end], label = "fitness")
plt.plot(x , binarity_list[start: end], label = "binarity")
plt.plot(x , grad_norm_normalized[start: end], label = "grad nrom normalized")
# plt.plot(x , grad_norm_L1[start: end], label = "grad nrom L1")

plt.xlabel("iteration number")
plt.ylabel("beta, fitness, binarity")
plt.ylabel("grad norm L1")

plt.legend(loc="upper left", framealpha = 0.2)
plt.xticks(np.arange(min(x), max(x) + 1, 10), rotation = 70, fontsize = 8) # for the x axis
plt.gca().set_xticks(np.arange(min(x), max(x) + 1, 1), minor = True)
plt.grid(which = "minor", alpha = 0.4)
plt.grid(which = "major", alpha = 0.8)
plt.savefig(f"beta_and_fitness_vs_iter_{start}_to_{end}.png", dpi = 500 )
plt.show()