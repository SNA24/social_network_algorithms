from joblib import Parallel, delayed
import networkx as nx
import itertools as it
import math
import matplotlib.pyplot as plt

def less(G, edge, directed = False):
    if directed:
        if G.out_degree(edge[0]) < G.out_degree(edge[1]):
            return 0
        if G.out_degree(edge[0]) == G.out_degree(edge[1]) and edge[0] < edge[1]:
            return 0
        return 1
    else:
        if G.degree(edge[0]) < G.degree(edge[1]):
            return 0
        if G.degree(edge[0]) == G.degree(edge[1]) and edge[0] < edge[1]:
            return 0
        return 1
    
def count_triangles_among_hh(triple, G):
    if G.is_directed():
        return (G.has_edge(triple[0], triple[1]) and G.has_edge(triple[1], triple[2]) and 
                G.has_edge(triple[2], triple[0]) or G.has_edge(triple[1], triple[0]) and 
                G.has_edge(triple[0], triple[2]) and G.has_edge(triple[2], triple[1]))
    else:
        return (G.has_edge(triple[0], triple[1]) and G.has_edge(triple[1], triple[2]) and 
                G.has_edge(triple[2], triple[0]))

def count_triangles_for_edge(G, edge, heavy_hitters):
    num_triangles = 0
    sel = less(G, edge, G.is_directed())
    if edge[sel] not in heavy_hitters:
        for u in G[edge[sel]]:
            if G.is_directed():
                if less(G, [u, edge[1 - sel]], G.is_directed()) and G.has_edge(edge[sel], edge[1 - sel]):
                    num_triangles += 1
            else:
                if less(G, [u, edge[1 - sel]], G.is_directed()) and G.has_edge(u, edge[1 - sel]):
                    num_triangles += 1
    return num_triangles

def num_triangles(G):

    num_triangles = 0

    m = nx.number_of_edges(G)
    heavy_hitters = set()

    heavy_hitters = {u for u in G.nodes() if G.degree(u) >= math.sqrt(m)}

    hh_triples = it.combinations(heavy_hitters, 3)
    for triple in hh_triples:
        if count_triangles_among_hh(triple, G):
            num_triangles += 1

    for edge in G.edges():
        num_triangles += count_triangles_for_edge(G, edge, heavy_hitters)

    return num_triangles

def num_triangles_parallel(G, j=2):

    num_triangles = 0

    m = nx.number_of_edges(G)
    heavy_hitters = set()

    heavy_hitters = {u for u in G.nodes() if G.degree(u) >= math.sqrt(m)}

    hh_triples = it.combinations(heavy_hitters, 3)
    triangles_among_hh = Parallel(n_jobs=j)(delayed(count_triangles_among_hh)(triple, G) for triple in hh_triples)
    num_triangles += sum(triangles_among_hh)

    edge_triangles = Parallel(n_jobs=j)(delayed(count_triangles_for_edge)(G, edge, heavy_hitters) for edge in G.edges())
    num_triangles += sum(edge_triangles)

    return num_triangles

if __name__ == '__main__':

    print("Undirected graph")
    G=nx.Graph()
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
    print("Parallel implementation", num_triangles_parallel(G, 4))

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
    print("Parallel implementation", num_triangles_parallel(G, 4))

    # Visualizzazione del grafo
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    labels = {edge: edge for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    plt.show()