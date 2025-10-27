# NeuroRouter: An AI-Powered Approach to Scalable Network Routing

![Language](https://img.shields.io/badge/Language-Python-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20%7C%20Matplotlib-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project presents a comparative analysis of classical graph-based routing algorithms versus a modern, AI-driven approach using Reinforcement Learning. The goal is to solve the critical scalability challenges inherent in dynamic network environments, using a simulated grid world that models real-world constraints like link latency and node failure.

The project demonstrates a clear evolution: from a baseline implementation of established academic algorithms (FCO/FPO) to an intelligent agent (Q-Learning) that achieves orders-of-magnitude better performance for real-time decision-making.

---

## Table of Contents

1.  [Project Overview](#project-overview)
2.  [The Core Problem: Scalability in Dynamic Networks](#the-core-problem)
3.  [Methodology: The Three Competing Algorithms](#methodology)
    - [Algorithm 1: FCO (Prim's Algorithm)](#algorithm-1-fco-prims-algorithm)
    - [Algorithm 2: FPO (Kruskal's Algorithm)](#algorithm-2-fpo-kruskals-algorithm)
    - [Algorithm 3: The AI Agent (Q-Learning)](#algorithm-3-the-ai-agent-q-learning)
4.  [The Benchmark: A Head-to-Head Comparison](#the-benchmark)
5.  [Results & Performance Analysis](#results--performance-analysis)
6.  [Project Structure](#project-structure)
7.  [How to Run This Project](#how-to-run-this-project)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
8.  [Key Learnings & Future Work](#key-learnings--future-work)
9.  [License](#license)
10. [Acknowledgments](#acknowledgments)

---

## <a name="project-overview"></a>1. Project Overview

Efficient routing in dynamic networks is a classic computer science challenge. Traditional algorithms, while mathematically robust, often struggle with performance as the network's size and complexity grow. This project explores this limitation by implementing and benchmarking two distinct paradigms for solving pathfinding problems:

1.  **Classical Algorithmic Approach:** Based on my research paper "UG-21CS204-209-244," this involves implementing Fuzzy Cut-Set Optimization (FCO) and Fuzzy Path Optimization (FPO), which are sophisticated applications of Prim's and Kruskal's algorithms, respectively. These methods calculate the optimal path (a Minimum Spanning Tree) based on a complete snapshot of the network's current state.

2.  **AI-Driven Approach:** To address the scalability issues of the classical methods, a **Reinforcement Learning** agent was developed from scratch. This agent is not given the rules of the network; instead, it *learns* an optimal navigation policy through trial-and-error in a simulated environment, developing an internal "knowledge base" (a Q-Table) of the best action to take from any given state.

The final benchmark quantitatively proves the superiority of the AI agent's real-time decision-making speed (inference), highlighting the power of shifting computational complexity from online calculation to an offline training phase.

## <a name="the-core-problem"></a>2. The Core Problem: Scalability in Dynamic Networks

The initial inspiration for this work came from the challenges of routing in **Underwater Acoustic Networks (UANs)**. These environments are characterized by high latency, limited energy resources, and constantly changing topologies. My original research focused on FCO and FPO, which use fuzzy logic to assign weights to network links based on factors like residual energy and signal strength.

However, these algorithms, even when optimized, share a fundamental limitation: their computational complexity. With a complexity of **O(E log E)**, their execution time grows significantly as the number of nodes (V) and edges (E) in the network increases. In a large-scale network requiring rapid route recalculations, this can become a critical bottleneck.

This project reframes that problem: **Can a system *learn* to route intelligently and make decisions in near-constant time, regardless of network size?**

## <a name="methodology"></a>3. Methodology: The Three Competing Algorithms

To create a fair and comprehensive comparison, three distinct algorithms were implemented in Python.

### <a name="algorithm-1-fco-prims-algorithm"></a>Algorithm 1: FCO (Prim's Algorithm)

This algorithm builds a Minimum Spanning Tree (MST) by growing a single tree from an arbitrary starting node.

-   **Strategy:** At each step, it finds the cheapest, most optimal edge that connects a node within the growing tree to a node outside of it.
-   **Implementation:** An efficient implementation using a min-heap (priority queue) was used to achieve a complexity of O(E log V).
-   **Analogy:** A city planner starting from a central point and always choosing the next best road to connect a new neighborhood.

### <a name="algorithm-2-fpo-kruskals-algorithm"></a>Algorithm 2: FPO (Kruskal's Algorithm)

This algorithm builds the MST by selecting the best edges from anywhere in the graph and connecting them, as long as they don't form a cycle.

-   **Strategy:** All edges in the graph are sorted by weight. The algorithm iterates through the sorted list, adding each edge to the final tree if it connects two previously unconnected components.
-   **Implementation:** A Disjoint Set Union (DSU) or "Union-Find" data structure was used for highly efficient cycle detection. The complexity is dominated by the initial sort at O(E log E).
-   **Analogy:** A regional contractor building the cheapest roads first, eventually linking all the separate segments into one connected network.

### <a name="algorithm-3-the-ai-agent-q-learning"></a>Algorithm 3: The AI Agent (Q-Learning)

This approach abandons explicit calculation in favor of learning through experience. The problem is modeled as a Reinforcement Learning environment.

-   **The Environment:** A 2D grid where `(0,0)` is the start and `(N-1, N-1)` is the destination. The grid contains obstacles (representing dead nodes) and high-cost zones (representing high latency links).
-   **The Agent:** A "packet" that learns to navigate the grid.
-   **State:** The agent's current `(row, col)` position on the grid.
-   **Actions:** The agent can move Up, Down, Left, or Right.
-   **Reward System:** The agent is trained using a carefully engineered reward function:
    -   **+100** for reaching the destination.
    -   **-50** for hitting an obstacle (a failed node).
    -   **-10** for stepping on a high-cost cell (a slow link).
    -   **-1** for every other move, to encourage finding the shortest path.
-   **Learning Algorithm:** The agent uses **Q-Learning** to build a Q-Table, which is essentially a "cheat sheet" mapping every state-action pair to a quality value. Over thousands of training episodes, it learns a policy that maximizes its cumulative reward, effectively discovering the optimal path on its own.

## <a name="the-benchmark"></a>4. The Benchmark

To compare performance, a benchmark script was created to measure the execution time of each algorithm across a range of network sizes.

-   **Test Case:** The algorithms were tested on randomly generated grids of increasing size (from 5x5 to 30x30).
-   **Measurement:** The script measures the `time.perf_counter()` to find a path from start to finish.
-   **Fairness:** For the ML Agent, only the **inference time** is measured—the time it takes to find the path *after* the one-time, offline training is complete. This simulates a real-world scenario where a pre-trained model is deployed for live decision-making.

## <a name="results--performance-analysis"></a>5. Results & Performance Analysis

The benchmark results starkly illustrate the performance differences between the two paradigms. The graph below plots execution time (Y-axis, logarithmic scale) against the number of nodes in the network (X-axis, logarithmic scale).

![Performance Comparison Graph](./performance_comparison.png)

### Analysis

-   **FCO (Prim's) & FPO (Kruskal's):** The execution time for the classical algorithms (blue and orange lines) increases at a clear polynomial rate. As the network grows larger, the time required to calculate the optimal path grows significantly, validating their computational complexity.
-   **ML Agent (Inference):** The execution time for the ML agent (green line) is several orders of magnitude lower and remains almost flat. This demonstrates a near **O(1) or constant-time** performance for decision-making. The agent's ability to simply look up the best action in its learned Q-Table makes it incredibly efficient and scalable for real-time applications.

**Conclusion:** The AI-driven approach successfully trades a one-time, offline training cost for exceptional online performance, making it a vastly more scalable solution for routing in large, dynamic networks.

## <a name="project-structure"></a>6. Project Structure

```
.
├── classic_algorithms.py     # Implementation of FCO (Prim's) and FPO (Kruskal's)
├── network_environment.py    # The grid world environment for the RL agent
├── q_learning_agent.py       # The core Q-Learning agent logic
├── main_trainer.py           # Script to train the agent and visualize its learned path
├── main_comparison.py        # Script to run the benchmark and generate the comparison graph
└── README.md                 # This file```

## <a name="how-to-run-this-project"></a>7. How to Run This Project

### <a name="prerequisites"></a>Prerequisites

-   Python 3.x
-   pip (Python package installer)

### <a name="installation"></a>Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
2.  Install the required libraries:
    ```bash
    pip install numpy matplotlib
    ```

### <a name="usage"></a>Usage

1.  **To train the AI agent and see its learned path:**
    Run the main trainer script. This will run the full training process and then display a plot of the optimal path found by the agent.
    ```bash
    python main_trainer.py
    ```

2.  **To run the performance benchmark and generate the comparison graph:**
    This script will time all three algorithms across various grid sizes and display the final performance plot.
    ```bash
    python main_comparison.py
    ```

## <a name="key-learnings--future-work"></a>8. Key Learnings & Future Work

This project was a deep dive into the practical trade-offs between classical algorithms and modern machine learning. Key takeaways include a strong understanding of computational complexity, the fundamentals of Reinforcement Learning (state, action, reward), and the importance of empirical benchmarking to validate performance claims.

**Potential Future Work:**
-   **Advanced RL Algorithms:** Implement more advanced algorithms like Deep Q-Learning (DQN) or A2C that can handle continuous or more complex state spaces.
-   **More Realistic Environment:** Evolve the grid world to include more dynamic elements, such as moving obstacles or links whose latency changes over time.
-   **Hyperparameter Tuning:** Systematically tune the learning rate, discount factor, and epsilon decay of the Q-Learning agent to optimize its training speed and final policy.

## <a name="license"></a>9. License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## <a name="acknowledgments"></a>10. Acknowledgments

-   The initial problem formulation was inspired by my paper **"UG-21CS204-209-244"**, which focused on optimizing FCO and FPO algorithms for underwater networks.
-   The foundational concepts of fuzzy logic and genetic algorithms were informed by the textbook **"Soft Computing: Fundamentals and Applications" by Dilip K. Pratihar**.