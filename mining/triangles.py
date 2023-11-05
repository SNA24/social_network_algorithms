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


def num_triangles(G, directed = False):
    num_triangles = 0
    m = nx.number_of_edges(G)

    heavy_hitters = set()
    for u in G.nodes():
        if G.degree(u) >= math.sqrt(m):
            heavy_hitters.add(u)

    for c in it.combinations(heavy_hitters, 3):
        if directed:
            for triple in it.permutations(c):
            # For undirected graphs, check if the triple forms a triangle.
                if G.has_edge(triple[0], triple[1]) and G.has_edge(triple[1], triple[2]) and G.has_edge(triple[2], triple[0]):
                    num_triangles += 1
            num_triangles /= 3
        else:        
            # For directed graphs, check if the triple forms a directed triangle.
            if G.has_edge(c[0], c[1]) and G.has_edge(c[1], c[2]) and G.has_edge(c[0], c[2]):
                num_triangles += 1
            
    for edge in G.edges():
        sel = less(G, edge,directed)
        if edge[sel] not in heavy_hitters:
            #arco tra il primo e il secondo nodo
            for u in G[edge[sel]]:
                if directed:
                    #arco tra il secondo e il terzo nodo
                    if less(G, [u, edge[1 - sel]], directed) and G.has_edge(edge[sel], edge[1 - sel]):
                        num_triangles += 1
                else:
                    if less(G, [u, edge[1 - sel]],directed) and G.has_edge(u, edge[1 - sel]):
                        num_triangles += 1
    return num_triangles


#parallel implementation for counting triangles
def count_triangle(G, triangle_candidates, directed = False):
    num_triangles = 0

    for c in triangle_candidates:
        if directed:
            for triple in it.permutations(c):
            # For undirected graphs, check if the triple forms a triangle.
                if G.has_edge(triple[0], triple[1]) and G.has_edge(triple[1], triple[2]) and G.has_edge(triple[2], triple[0]):
                    num_triangles += 1
            num_triangles /= 3
        else:        
            # For directed graphs, check if the triple forms a directed triangle.
            if G.has_edge(c[0], c[1]) and G.has_edge(c[1], c[2]) and G.has_edge(c[0], c[2]):
                num_triangles += 1
    return num_triangles

def num_triangles_parallel(G, j, directed = False):
    num_triangles = 0

    m = nx.number_of_edges(G)
    heavy_hitters = set()

    for u in G.nodes():
        if G.degree(u) >= math.sqrt(m):
            heavy_hitters.add(u)

    triangle_candidates = list(it.combinations(heavy_hitters, 3))
    size = math.ceil(len(list(triangle_candidates))/j)
    # Each process counts the triangles in a subset of triangle_candidates
    with Parallel(n_jobs = j) as parallel:
        # Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        
        #print(triangle_candidates[i:i+size] for i in range(j))
        result = parallel(delayed(count_triangle)(G, triangle_candidates[i:i+size],directed) for i in range(j))
        # Aggregates the results
        num_triangles = sum(result)

    for edge in G.edges():
        sel = less(G, edge,directed)
        if edge[sel] not in heavy_hitters:
            for u in G[edge[sel]]:
                if directed:
                    '''print(edge[sel], u, edge[1 - sel])
                    print(less(G, [u, edge[1 - sel]], directed))
                    print(G.has_edge(edge[sel], edge[1 - sel]))'''
                    if less(G, [u, edge[1 - sel]], directed) and G.has_edge(edge[sel], edge[1 - sel]):
                        num_triangles += 1
                else:
                    if less(G, [u, edge[1 - sel]],directed) and G.has_edge(u, edge[1 - sel]):
                        num_triangles += 1
    return num_triangles



if __name__ == '__main__':
    print("GRAFO DIRETTO")
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

    
    triang = num_triangles(G,directed=True)
    print("Naive implementation")
    print(triang)

    print("Parallel implementation")
    print(num_triangles_parallel(G,2,directed=True))

    # Visualizzazione del grafo
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    labels = {edge: edge for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    plt.show()