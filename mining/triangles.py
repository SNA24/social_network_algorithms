from joblib import Parallel, delayed
import networkx as nx
import itertools as it
import math
import matplotlib.pyplot as plt
import multiprocessing as mp

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilities.parallel_algorithms import degree, split_list

#Optimized algorithm
#There are two optimizations.
#
#OPTIMIZATION1: It consider an order among nodes. Specifically, nodes are ordered by degree. In case of nodes with the same degree, nodes are ordered by label.
#In this way a triangle is counted only once. Specifically, from the node with smaller degree to the one with larger degree.
def less(G, edge):
    if degree(G, edge[0]) < degree(G, edge[1]): 
        return 0
    if degree(G, edge[0]) == degree(G, edge[1]) and edge[0] < edge[1]:
        return 0
    return 1

#OPTIMIZTION2: It distinguishes between high-degree nodes (called heavy hitters) and low-degree nodes.
#Triangles involving only heavy hitters (that have been recognized to be the bottleneck of the naive algorithm) are handled in a different way respect to remaining triangles.
def num_triangles(G):
    num_triangles = 0
    m = nx.number_of_edges(G)

    # The set of heavy hitters, that is nodes with degree at least sqrt(m)
    # Note: the set contains at most sqrt(m) nodes, since num_heavy_hitters*sqrt(m) must be at most the sum of degrees = 2m
    # Note: the choice of threshold sqrt(m) is the one that minimize the running time of the algorithm.
    # A larger value of the threshold implies a faster processing of triangles containing only heavy hitters, but a slower processing of remaining triangles.
    # A smaller value of the threshold implies the reverse.
    heavy_hitters=set()
    for u in G.nodes():
        if degree(G,u) >= math.sqrt(m):
            heavy_hitters.add(u)

    # Number of triangles among heavy hitters.
    # It considers all possible triples of heavy hitters, and it verifies if it forms a triangle.
    # The running time is then O(sqrt(m)^3) = m*sqrt(m)
    for triple in it.combinations(heavy_hitters,3):
        if G.has_edge(triple[0], triple[1]) and G.has_edge(triple[1], triple[2]) and G.has_edge(triple[2], triple[0]):
            num_triangles += 1
        if G.is_directed():
            if G.has_edge(triple[1], triple[0]) and G.has_edge(triple[2], triple[1]) and G.has_edge(triple[0], triple[2]):
                num_triangles += 1

    found = {node: dict() for node in G.nodes()}

    # Number of remaining triangles.
    # For each edge, if one of the endpoints is not an heavy hitter, verifies if there is a node in its neighborhood that forms a triangle with the other endpoint.
    # This is essentially the naive algorithm optimized to count only ordered triangle in which the first vertex (i.e., u) is not an heavy hitter.
    # Since the size of the neighborhood of a non heavy hitter is at most sqrt(m), the complexity is O(m*sqrt(m))

    for edge in G.edges():
        sel = less(G,edge)
        if edge[sel] not in heavy_hitters:
            if not G.is_directed():
                for u in G[edge[sel]]:
                    if less(G,[u,edge[1-sel]]) and G.has_edge(u,edge[1-sel]):
                        num_triangles += 1
            else:
                # consider predecessors
                for u in G.predecessors(edge[sel]):
                    if G.has_edge(edge[1-sel],u):
                        triangle = tuple(sorted([u,edge[1-sel],edge[sel]]))
                        if triangle not in found.keys():
                            found[triangle] = 1
                            num_triangles += 1
                # consider successors
                for u in G.successors(edge[sel]):
                    if G.has_edge(u,edge[1-sel]):
                        triangle = tuple(sorted([u,edge[1-sel],edge[sel]]))
                        if triangle not in found.keys():
                            found[triangle] = 1
                            num_triangles += 1

    return num_triangles

def find_triangles_in_HH(G, combinations):

    num_triangles = 0

    for triple in combinations: 
        if G.has_edge(triple[0], triple[1]) and G.has_edge(triple[1], triple[2]) and G.has_edge(triple[2], triple[0]):
            num_triangles += 1
        if G.is_directed():
            if G.has_edge(triple[1], triple[0]) and G.has_edge(triple[2], triple[1]) and G.has_edge(triple[0], triple[2]):
                num_triangles += 1

    return num_triangles

def parallel_num_triangles(G, n_jobs=4):
    
    num_triangles = 0
    m = nx.number_of_edges(G)

    heavy_hitters = set()
    for u in G.nodes():
        if degree(G, u) >= math.sqrt(m):
            heavy_hitters.add(u)

    combinations = list(it.combinations(heavy_hitters, 3))
    found = mp.Manager().dict()

    # Function to find triangles among heavy hitters
    def find_triangles_in_HH_wrapper(combo):
        return find_triangles_in_HH(G, combo)
    
    # Parallel computation for heavy hitters
    combinations = list(split_list(combinations, math.ceil(len(combinations) / n_jobs)))
    num_triangles += sum(Parallel(n_jobs=n_jobs)(delayed(find_triangles_in_HH_wrapper)(combo) for combo in combinations))

    for edge in G.edges():
        sel = less(G,edge)
        if edge[sel] not in heavy_hitters:
            if not G.is_directed():
                for u in G[edge[sel]]:
                    if less(G,[u,edge[1-sel]]) and G.has_edge(u,edge[1-sel]):
                        num_triangles += 1
            else:
                # consider predecessors
                for u in G.predecessors(edge[sel]):
                    if G.has_edge(edge[1-sel],u):
                        triangle = tuple(sorted([u,edge[1-sel],edge[sel]]))
                        if triangle not in found.keys():
                            found[triangle] = 1
                            num_triangles += 1
                # consider successors
                for u in G.successors(edge[sel]):
                    if G.has_edge(u,edge[1-sel]):
                        triangle = tuple(sorted([u,edge[1-sel],edge[sel]]))
                        if triangle not in found.keys():
                            found[triangle] = 1
                            num_triangles += 1

    return num_triangles

if __name__ == '__main__':

    print("Undirected graph")
    G=nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('C', 'A')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'A')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'D')

    print("Optimized implementation", num_triangles(G))
    print("Parallel implementation", parallel_num_triangles(G, 4))

    print("Directed graph")
    G=nx.DiGraph()
    G.add_edge('A', 'B')
    G.add_edge('C', 'A')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'A')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'D')
    
    print("Optimized implementation", num_triangles(G))
    print("Parallel implementation", parallel_num_triangles(G, 4))

    # Visualizzazione del grafo
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    labels = {edge: edge for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    plt.show()