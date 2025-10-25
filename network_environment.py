# network_environment.py
import numpy as np

class NetworkGrid:
    def __init__(self, grid_size=10):
        self.size = grid_size
        # Define the layout of our "underwater network"
        self.start_node = (0, 0)
        self.end_node = (grid_size - 1, grid_size - 1)
        
        # These are like nodes with zero energy
        self.dead_nodes = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 7), (7, 6)]
        
        # These are like nodes with high resource utilization (slow paths)
        self.slow_links = [(1, 5), (4, 8), (8, 4)]
        
        self.agent_position = self.start_node

    def reset(self):
        """Called at the start of every new attempt (episode)."""
        self.agent_position = self.start_node
        return self.agent_position

    def get_reward(self):
        """Calculates the reward for the agent's current position."""
        if self.agent_position == self.end_node:
            return 100  # High reward for reaching the destination
        if self.agent_position in self.dead_nodes:
            return -50  # High penalty for hitting a dead node
        if self.agent_position in self.slow_links:
            return -10  # Moderate penalty for taking a slow path
        return -1       # Small penalty for every move to encourage speed

    def step(self, action):
        """
        The agent makes a move.
        Actions are: 0=Up, 1=Down, 2=Left, 3=Right
        """
        row, col = self.agent_position
        if action == 0: row = max(row - 1, 0)
        elif action == 1: row = min(row + 1, self.size - 1)
        elif action == 2: col = max(col - 1, 0)
        elif action == 3: col = min(col + 1, self.size - 1)
        self.agent_position = (row, col)

        reward = self.get_reward()
        
        # An attempt is 'done' if it reaches the end or hits a dead node
        done = self.agent_position == self.end_node or self.agent_position in self.dead_nodes
        
        return self.agent_position, reward, done