import networkx as nx
import matplotlib.pyplot as plt
import math
import itertools as it
from joblib import Parallel, delayed

#Classic algorithm for computing Diameter with directed and undirected networks
def diameter(G, sample=None, directed=False):
    nodes = G.nodes()
    n = len(nodes)
    diam = 0
    if sample is None:
        sample = nodes

    visited = set()
    
    for u in sample:
        udiam = 0
        clevel = [u]
        visited.add(u)
        while len(visited) < n:
            nlevel = []
            while len(clevel) > 0:
                c = clevel.pop()
                for v in G.neighbors(c) if directed else G[c]:
                    if v not in visited:
                        visited.add(v)
                        nlevel.append(v)
            clevel = nlevel
            udiam += 1
        if udiam > diam:
            diam = udiam

    return diam

#PARALLEL IMPLEMENTATION - CONTROLLARE PER I GRAFI DIRETTI
#Utility used for split a vector data in chunks of the given size.
def chunks(data, size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

#Parallel implementation of diameter with joblib
def parallel_diam(G,j,directed=False):
    diam = 0

    # Initialize the class Parallel with the number of available process
    with Parallel(n_jobs = j) as parallel:
        # Run in parallel diameter function on each processor by passing to each processor only the subset of nodes on which it works
        result = parallel(delayed(diameter)(G, X, directed) for X in chunks(G.nodes(),math.ceil(len(G.nodes())/j)))
        # Aggregates the results
        diam = max(result)

    return diam

#AD-HOC OPTIMIZATION for directed and undirected networks
def stream_diam(G, directed=False):
    step = 0

    # At the beginning, R contains for each vertex v the number of nodes that can be reached from v in one step
    R = {v: G.degree(v) if not directed else (G.in_degree(v), G.out_degree(v)) for v in G.nodes()}
    print(R)
    done = False

    while not done:
        done = True
        for edge in G.edges():
            if directed:
                if R[edge[0]] != R[edge[1]]:
                    in_degree_0, out_degree_0 = R[edge[0]]
                    in_degree_1, out_degree_1 = R[edge[1]]
                    R[edge[0]] = (max(in_degree_0, in_degree_1), max(out_degree_0, out_degree_1))
                    R[edge[1]] = R[edge[0]]
                    done = False
            else:
                if R[edge[0]] != R[edge[1]]:
                    R[edge[0]] = max(R[edge[0]], R[edge[1]])
                    R[edge[1]] = R[edge[0]]
                    done = False
        step += 1

    return step




if __name__ == '__main__':
    '''G=nx.Graph()
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'F')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'G')'''
    #print(diameter(G))

    # Creazione di un grafo diretto strettamente connesso
    G = nx.DiGraph()

    # Aggiunta di nodi
    G.add_node('A')
    G.add_node('B')
    G.add_node('C')
    G.add_node('D')
    G.add_node('E')

    # Collegamento dei nodi in una catena
    G.add_edge('A', 'B')
    G.add_edge('B', 'C')
    G.add_edge('C', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'A')
    G.add_edge('E','A')
    

    # Calculate the diameter of the directed graph
    '''diam = diameter(G, directed=True)
    print(diam)'''

    
    print(stream_diam(G,directed=True))

    # Visualizzazione del grafo
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    labels = {edge: edge for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    plt.show()

    

