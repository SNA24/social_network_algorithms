import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import networkx as nx
import math
import random
import networkx as nx
from joblib import Parallel, delayed
from utilities.parallel_algorithms import chunks
# Computes edge and vertex betweenness of the graph in input
# The algorithm is quite time-consuming. Indeed, its computational complexity is O(nm).
# Possible optimizations: parallelization, sampling

def betweenness(G, sample=None):

    if sample is None:
        sample = G.nodes()

    # Initialize the betweenness of each edge and vertex to 0 in sample
    edge_btw = {frozenset(e):0 for e in G.edges()}
    node_btw = {i:0 for i in G.nodes()}

    for u in sample:
        # Compute the number of shortest paths from u to every other node
        tree = [] # It lists the nodes in the order in which they are visited
        spnum = {i:0 for i in G.nodes()} # It saves the number of shortest paths from u to i
        parents = {i:[] for i in G.nodes()} # It saves the parents of i in each of the shortest paths from u to i
        distance = {i:-1 for i in G.nodes()} # The length of the shortest path from u to i
        eflow = {frozenset(e):0 for e in G.edges()} # The number of shortest paths from u to other nodes that use the edge e
        vflow = {i:1 for i in G.nodes()} # The number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        #BFS
        queue = [u]
        spnum[u] = 1
        distance[u] = 0
        while len(queue) > 0:
            c = queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1: # If vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c] + 1
                if distance[i] == distance[c] + 1: # If we have found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        #BOTTOM-UP PHASE
        while len(tree) > 0:
            c = tree.pop()
            for i in parents[c]:
                e=frozenset((c,i))
                eflow[e] += vflow[c] * (spnum[i]/spnum[c]) # The number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[e] # Each shortest path that use an edge (i,c) where i is closest to u than c must use also vertex i
                edge_btw[e] += eflow[e] # Betweenness of an edge is the sum over all sources u of the number of shortest paths from u to other nodes using that edge
            if c != u:
                node_btw[i] += vflow[i] # Betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw,node_btw

# PARALLELIZATION
def parallel_betweenness(G, j=1):

    # Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        # run in parallel the betweenness function on each job by passing to each job only the subset of nodes on which it works
        result = parallel(delayed(betweenness)(G, X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

        # Aggregates the results
        edge_btw = {frozenset(e):0 for e in G.edges()}
        node_btw = {i:0 for i in G.nodes()}

        for r in result:
            for e in r[0]:
                edge_btw[e] += r[0][e]
            for i in r[1]:
                node_btw[i] += r[1][i]

    return edge_btw, node_btw

def sampled_betweenness(G, sample_size=20000, partition=None):

    if partition is None:
        partition = G.nodes()
        
    if len(partition) < sample_size:
        sample_size = len(partition)
        
    
    resample = random.sample(list(partition), sample_size)

    return betweenness(G, resample)

def parallel_sampled_betweenness(G, sample_size=20000, j=1):

    # Initialize the class Parallel with the number of available process
    with Parallel(n_jobs=j) as parallel:
        # run in parallel the betweenness function on each job by passing to each job only the subset of nodes on which it works
        result = parallel(delayed(sampled_betweenness)(G, sample_size, X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

        # Aggregates the results
        edge_btw = {frozenset(e):0 for e in G.edges()}
        node_btw = {i:0 for i in G.nodes()}

        for r in result:
            for e in r[0]:
                edge_btw[e] += r[0][e]
            for i in r[1]:
                node_btw[i] += r[1][i]

    return edge_btw, node_btw

if __name__ == '__main__':
    G = nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'F')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'G')

    print("betweenness")
    print(sorted(betweenness(G)[1].items(), key=lambda x: x[1], reverse=True))
    print("parallel_betweenness")
    print(sorted(parallel_betweenness(G)[1].items(), key=lambda x: x[1], reverse=True))

    G = nx.DiGraph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'F')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'G')

    print("betweenness")
    print(sorted(betweenness(G)[1].items(), key=lambda x: x[1], reverse=True))
    print("parallel_betweenness")
    print(sorted(parallel_betweenness(G)[1].items(), key=lambda x: x[1], reverse=True))