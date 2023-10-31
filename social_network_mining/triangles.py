import networkx as nx
import itertools as it
import math
import matplotlib.pyplot as plt


def less(G, edge):
    if G.out_degree(edge[0]) < G.out_degree(edge[1]):
        return 0
    if G.out_degree(edge[0]) == G.out_degree(edge[1]) and edge[0] < edge[1]:
        return 0
    return 1

def num_triangles(G):
    num_triangles = 0
    m = nx.number_of_edges(G)
    heavy_hitters=set()
    for u in G.nodes():
        if G.degree(u) >= math.sqrt(m):
            heavy_hitters.add(u)

    for triple in it.combinations(heavy_hitters,3):
        for c in it.permutations(triple):
            if G.has_edge(c[0], c[1]) and G.has_edge(c[1], c[2]) and G.has_edge(c[2], c[0]):
                num_triangles += 1
        num_triangles /= 3
    for edge in G.edges():
        sel = less(G,edge)
        if edge[sel] not in heavy_hitters:
            for u in G.successors(edge[sel]):
                if less(G,[u,edge[1-sel]]) and G.has_edge(u,edge[1-sel]):
                    num_triangles += 1
                    print(num_triangles)

    return num_triangles



if __name__ == '__main__':
    '''G=nx.DiGraph()
    G.add_edge('A', 'B')
    G.add_edge('C', 'A')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'A')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'D')'''

    # Crea un grafo diretto
    G = nx.DiGraph()

    # Aggiungi nodi
    G.add_node('A')
    G.add_node('B')
    G.add_node('C')
    G.add_node('D')

    # Aggiungi archi
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    G.add_edge('C', 'A')
    G.add_edge('B', 'D')
    
    triang = num_triangles(G)

    print(triang)

    # Visualizzazione del grafo
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    labels = {edge: edge for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    plt.show()