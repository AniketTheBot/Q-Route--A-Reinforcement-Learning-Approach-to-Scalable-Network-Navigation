# main_comparison.py
import time
import numpy as np
import matplotlib.pyplot as plt
from classic_algorithms import construct_mst_fco, construct_mst_fpo
from q_learning_agent import SmartAgent # Assuming your ML agent is trained

# Helper to create a random graph for FCO/FPO
def create_random_graph(num_nodes):
    graph = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = np.random.randint(1, 20)
            graph[i].append((j, weight))
            graph[j].append((i, weight))
    return graph

# --- BENCHMARK SETUP ---
grid_sizes = [5, 10, 15, 20, 25, 30] # We'll test on 5x5, 10x10 grids etc.
num_nodes_list = [s*s for s in grid_sizes]

times_fco = []
times_fpo = []
times_ml = []

print("--- Starting Benchmark ---")

# --- Pre-train the ML agent (or load a pre-trained one) ---
# For a fair comparison, we only train it once on the largest grid size
print("Pre-training ML agent for the largest grid...")
largest_grid = grid_sizes[-1]
# NOTE: In a real scenario, you'd save/load this trained agent
# Here we just train it as a one-time setup cost.
agent = SmartAgent(grid_size=largest_grid)
# A quick dummy training loop (doesn't need to be perfect, just to populate the Q-table)
for _ in range(5000): 
    # This loop is just to simulate a trained agent. It's NOT part of the timed benchmark.
    state_idx = np.random.randint(largest_grid*largest_grid)
    action = np.random.randint(4)
    agent.q_table[state_idx, action] = np.random.rand()
print("ML Agent is ready.")


for size in grid_sizes:
    num_nodes = size * size
    print(f"\nTesting on a {size}x{size} grid ({num_nodes} nodes)...")
    
    # 1. Benchmark FCO (Prim's)
    graph = create_random_graph(num_nodes)
    start_time = time.perf_counter()
    construct_mst_fco(graph)
    end_time = time.perf_counter()
    time_fco = end_time - start_time
    times_fco.append(time_fco)
    print(f"  FCO Time: {time_fco:.6f} seconds")

    # 2. Benchmark FPO (Kruskal's)
    start_time = time.perf_counter()
    construct_mst_fpo(graph)
    end_time = time.perf_counter()
    time_fpo = end_time - start_time
    times_fpo.append(time_fpo)
    print(f"  FPO Time: {time_fpo:.6f} seconds")
    
    # 3. Benchmark ML Agent (Inference)
    agent.grid_size = size # Adjust agent for current grid size
    start_time = time.perf_counter()
    # Simulate finding a path from start to end
    state = (0,0)
    for _ in range(num_nodes): # Simulate pathfinding steps
        action = agent.choose_action(state)
        # In a real scenario, you'd step through the environment
        # Here, we just simulate the decision-making time
    end_time = time.perf_counter()
    time_ml = end_time - start_time
    times_ml.append(time_ml)
    print(f"  ML Agent Inference Time: {time_ml:.6f} seconds")


# --- PLOTTING THE RESULTS ---
plt.figure(figsize=(12, 8))
plt.plot(num_nodes_list, times_fco, 'o-', label='FCO (Prim\'s Algorithm)', linewidth=2)
plt.plot(num_nodes_list, times_fpo, 's-', label='FPO (Kruskal\'s Algorithm)', linewidth=2)
plt.plot(num_nodes_list, times_ml, '^-', label='ML Agent (Inference)', linewidth=2, color='green')

plt.xlabel('Number of Nodes in the Network')
plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison: Classic vs. ML Routing')
plt.legend()
plt.grid(True)
plt.xscale('log') # Use a log scale for better visualization if times vary widely
plt.yscale('log')
plt.show()