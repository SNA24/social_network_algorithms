import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

import math
from utilities.priorityq import PriorityQueue
from utilities.parallel_algorithms import chunks
import networkx as nx
from joblib import Parallel, delayed

def hierarchical(G):
    # Create a priority queue with each pair of nodes indexed by distance
    pq = PriorityQueue()
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                if G.has_edge(u, v):
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 0)
                else:
                    pq.add(frozenset([frozenset(u), frozenset(v)]), 1)

    # Start with a cluster for each node
    clusters = set(frozenset(u) for u in G.nodes())
    # This is one of the measures of quality of a partition implemented in networkx
    # Other measures are returned by nx.algorithms.community.partition_quality(G, clusters)
    # Please read the documentation for more information
    q = nx.algorithms.community.modularity(G, clusters)

    done = False
    while not done:
        # Merge closest clusters
        s = list(pq.pop())
        clusters.remove(s[0])
        clusters.remove(s[1])

        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset([s[0] | s[1], w]), 0)
            else:
                pq.add(frozenset([s[0] | s[1], w]), 1)

        clusters.add(s[0] | s[1])

        newq = nx.algorithms.community.modularity(G, clusters)
        if abs(newq) > abs(q):
            done = True
        else:
            q = newq

    return clusters

def create_priority_queue(G, nodes):
    pq = []
    for u in nodes:
        for v in G.nodes():
            if u != v:
                if G.has_edge(u, v):
                    pq.append((frozenset([frozenset(u), frozenset(v)]), 0))
                else:
                    pq.append((frozenset([frozenset(u), frozenset(v)]), 1))
    return pq

def parallel_hierarchical(G, num_jobs=8):
    nodes = list(G.nodes())
    num_nodes = len(nodes)

    with Parallel(n_jobs=num_jobs) as parallel:
        results = parallel(delayed(create_priority_queue)(G, nodes) for nodes in chunks(G.nodes(),math.ceil(len(G.nodes())/num_jobs)))

    pq = PriorityQueue()
    for result in results:
        for elem in result:
            pq.add(elem[0], elem[1])

    # Start with a cluster for each node
    clusters = set(frozenset(u) for u in G.nodes())
    # This is one of the measures of quality of a partition implemented in networkx
    # Other measures are returned by nx.algorithms.community.partition_quality(G, clusters)
    # Please read the documentation for more information
    q = nx.algorithms.community.modularity(G, clusters)

    done = False
    while not done:
        # Merge closest clusters
        s = list(pq.pop())
        
        clusters.remove(s[0])
        clusters.remove(s[1])

        # Update the distance of other clusters from the merged cluster
        for w in clusters:
            e1 = pq.remove(frozenset([s[0], w]))
            e2 = pq.remove(frozenset([s[1], w]))
            if e1 == 0 or e2 == 0:
                pq.add(frozenset([s[0] | s[1], w]), 0)
            else:
                pq.add(frozenset([s[0] | s[1], w]), 1)

        clusters.add(s[0] | s[1])

        newq = nx.algorithms.community.modularity(G, clusters)
        if abs(newq) > abs(q):
            done = True
        else:
            q = newq

    return clusters

if __name__ == '__main__':

    print('Unidrected graph')

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

    print("Hierarchical clustering", hierarchical(G))
    print("Parallel hierarchical clustering", parallel_hierarchical(G)) 
    
    print('Directed graph')

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

    print("Hierarchical clustering", hierarchical(G))
    print("Parallel hierarchical clustering", parallel_hierarchical(G))
