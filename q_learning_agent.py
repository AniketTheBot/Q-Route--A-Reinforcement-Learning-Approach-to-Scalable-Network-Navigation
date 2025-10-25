# q_learning_agent.py
import numpy as np

class SmartAgent:
    def __init__(self, grid_size, num_actions=4):
        # The Q-Table is our agent's brain.
        # Rows = each possible position on the grid.
        # Cols = each possible action (Up, Down, Left, Right).
        self.q_table = np.zeros((grid_size * grid_size, num_actions))
        self.num_actions = num_actions
        self.grid_size = grid_size

        # --- ML Hyperparameters ---
        self.learning_rate = 0.1      # How fast the agent learns.
        self.discount_factor = 0.99   # How much it values future rewards.
        self.epsilon = 1.0            # Exploration Rate: 1.0 = 100% random moves.
        self.epsilon_decay = 0.9995   # We slowly reduce exploration over time.
        self.min_epsilon = 0.01       # Minimum exploration rate.

    def state_to_index(self, state):
        """Helper function to turn a (row, col) tuple into a single table row index."""
        return state[0] * self.grid_size + state[1]

    def choose_action(self, state):
        """
        Decide whether to explore randomly or use the Q-Table to make the best move.
        This is called the Epsilon-Greedy strategy.
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.num_actions)  # Explore: Take a random action
        else:
            state_index = self.state_to_index(state)
            return np.argmax(self.q_table[state_index]) # Exploit: Take the best known action

    def learn(self, state, action, reward, next_state):
        """
        This is the most important function. It updates the Q-Table after each move.
        This is where the "learning" happens!
        """
        state_index = self.state_to_index(state)
        next_state_index = self.state_to_index(next_state)

        old_value = self.q_table[state_index, action]
        next_max_value = np.max(self.q_table[next_state_index])

        # The Q-Learning formula
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max_value - old_value)
        self.q_table[state_index, action] = new_value

        # Decay epsilon so the agent explores less as it gets smarter
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay