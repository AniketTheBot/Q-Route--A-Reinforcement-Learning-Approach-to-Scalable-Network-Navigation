# main_trainer.py
from network_environment import NetworkGrid
from q_learning_agent import SmartAgent
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETUP THE SIMULATION ---
GRID_SIZE = 10
env = NetworkGrid(grid_size=GRID_SIZE)
agent = SmartAgent(grid_size=GRID_SIZE)

# --- 2. TRAIN THE AGENT ---
num_episodes = 20000  # How many times the agent will try to solve the maze
for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Agent chooses an action
        action = agent.choose_action(state)
        
        # Environment responds with the next state, reward, and whether the attempt is done
        next_state, reward, done = env.step(action)
        
        # Agent learns from the experience
        agent.learn(state, action, reward, next_state)
        
        # Move to the next state
        state = next_state

    if (episode + 1) % 1000 == 0:
        print(f"Episode: {episode + 1}/{num_episodes} | Epsilon: {agent.epsilon:.4f}")

print("The Model is trained !!")

# --- 3. VISUALIZE THE LEARNED PATH ---
# Now we test the agent with exploration turned off (epsilon = 0)
state = env.reset()
path = [state]
done = False
agent.epsilon = 0 # Turn off exploration for the final test run

while not done and len(path) < 50: # Add a step limit to prevent infinite loops
    action = agent.choose_action(state)
    state, reward, done = env.step(action)
    path.append(state)

print(f"Path found by agent: {path}")

# Plotting code
grid_to_plot = np.zeros((GRID_SIZE, GRID_SIZE))
for node in env.dead_nodes: grid_to_plot[node] = 1 # Mark dead nodes
for node in env.slow_links: grid_to_plot[node] = 0.5 # Mark slow links

plt.figure(figsize=(8, 8))
plt.imshow(grid_to_plot, cmap='gist_gray_r', interpolation='nearest')

path_rows, path_cols = zip(*path)
plt.plot(path_cols, path_rows, marker='o', markersize=8, linewidth=3, color='blue', label='Learned Path')
plt.plot(env.start_node[1], env.start_node[0], marker='s', markersize=15, color='green', label='Start')
plt.plot(env.end_node[1], env.end_node[0], marker='*', markersize=20, color='red', label='End')

plt.title('Learned Path by the Smart Routing Agent')
plt.xticks(np.arange(GRID_SIZE))
plt.yticks(np.arange(GRID_SIZE))
plt.grid(True)
plt.legend()
plt.show()