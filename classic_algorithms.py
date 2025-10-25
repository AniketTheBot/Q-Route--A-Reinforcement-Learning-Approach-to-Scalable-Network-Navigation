# classic_algorithms.py
import heapq

# FCO (Prim's Algorithm)
def construct_mst_fco(graph):
    num_nodes = len(graph)
    if num_nodes == 0:
        return 0, []

    visited = [False] * num_nodes
    mst_cost = 0
    mst_edges = []
    
    # Min-heap (priority queue) to store edges
    # We use a more efficient version than your original O(V^2) implementation
    pq = [(0, 0, -1)]  # (weight, current_node, from_node)

    while pq:
        weight, u, parent = heapq.heappop(pq)

        if visited[u]:
            continue

        visited[u] = True
        mst_cost += weight
        if parent != -1:
            mst_edges.append((parent, u))
        
        if all(visited):
            break

        for v, w in graph[u]:
            if not visited[v]:
                heapq.heappush(pq, (w, v, u))
    
    return mst_cost

# FPO (Kruskal's Algorithm)
class DSU: # Disjoint Set Union (for cycle detection)
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def construct_mst_fpo(graph):
    num_nodes = len(graph)
    edges = []
    for u in range(num_nodes):
        for v, w in graph[u]:
            edges.append((w, u, v))
    
    edges.sort() # Sort all edges by weight
    
    dsu = DSU(num_nodes)
    mst_cost = 0
    mst_edges = []
    
    for weight, u, v in edges:
        if dsu.union(u, v):
            mst_cost += weight
            mst_edges.append((u, v))
            if len(mst_edges) == num_nodes - 1:
                break
                
    return mst_cost