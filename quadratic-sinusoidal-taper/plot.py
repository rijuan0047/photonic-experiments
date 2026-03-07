import json
import matplotlib.pyplot as plt
import numpy as np

with open("history.json", "r") as f:
    data = json.load(f)

generations = []
best_scores = []
avg_scores = []

a = []
b = []
c = []

for gen in data:
    g = int(gen["generation"])
    generations.append(g)

    scores = [s["score"] for s in gen["solutions"]]

    best_scores.append(max(np.abs(scores)))   
    avg_scores.append(np.mean(np.abs(scores)))

    best_sol = min(gen["solutions"], key=lambda x: x["score"])
    p = best_sol["params"]

    a.append(p[0])
    b.append(p[1])
    c.append(p[2])

plt.figure()
plt.plot(generations, best_scores, label="best bcore")
plt.plot(generations, avg_scores, label="average score")
plt.xlabel("generation")
plt.ylabel("score")
plt.title("optimization progress")
plt.legend()
plt.grid(True)
plt.savefig("scores-vs-generation.png", dpi = 300)

plt.figure()
plt.plot(generations, a, label="a")
plt.plot(generations, b, label="b")
plt.plot(generations, c, label="c")
plt.xlabel("generation")
plt.ylabel("parameter value")
plt.title("parameter evolution")
plt.legend()
plt.grid(True)
plt.savefig("parameters-vs-generation.png", dpi = 300)

plt.show()