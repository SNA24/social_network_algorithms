import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed
from utilities.parallel_algorithms import chunks, neighbors

# The measure associated to each node is the sum of the (shortest) distances of this node from each remaining node
# It is not exactly the closeness measure, but it returns the same ranking on vertices
def closeness(G, sample=None):

    if sample is None:
        sample = G.nodes()

    cen=dict()

    for u in sample:
        visited=set()
        visited.add(u)
        queue = [u]
        dist = dict()
        dist[u]  = 0
        while queue != []:
            v = queue.pop(0)
            for w in neighbors(G, v):
                if w not in visited:
                    queue.append(w)
                    visited.add(w)
                    dist[w] = dist[v] + 1
        cen[u] = sum(dist.values())

    return cen

def parallel_closeness(G, j):

    cen = {}

    with Parallel(n_jobs=j) as parallel:
        results = parallel(delayed(closeness)(G, sample) for sample in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

        for i in results:
            cen.update(i)

    return cen

if __name__ == '__main__':
        
    print("Undirected graph")
    G=nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'F')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'G')

    # for each closeness measure, order the list by value and print the keys
    print("Closeness measure")
    print(sorted(closeness(G).items(), key=lambda x: x[1], reverse=True))
    print("Parallel closeness measure")
    print(sorted(parallel_closeness(G, 2).items(), key=lambda x: x[1], reverse=True))

    print("Directed graph")
    G=nx.DiGraph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'F')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'G')

    # for each closeness measure, order the list by value and print the keys
    print("Closeness measure")
    print(sorted(closeness(G).items(), key=lambda x: x[1], reverse=True))
    print("Parallel closeness measure")
    print(sorted(parallel_closeness(G, 2).items(), key=lambda x: x[1], reverse=True))