import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from joblib import Parallel, delayed
import networkx as nx
import random
import math
from utilities.parallel_algorithms import chunks, neighbors

def two_means(G):
    n=G.number_of_nodes()
    # Choose two clusters represented by vertices that are not neighbors
    u = random.choice(list(G.nodes()))
    v = random.choice(list(nx.non_neighbors(G, u)))
    cluster0 = {u}
    cluster1 = {v}
    added = 2

    while added < n:
        # Choose a node that is not yet in a cluster and add it to the closest cluster
        list_nodes = [el for el in G.nodes() if el not in cluster0|cluster1 and (len(
            neighbors(G,el).intersection(cluster0)) != 0 or len(neighbors(G,el).intersection(cluster1)) != 0)]
        if len(list_nodes) <= 0:
            break
        x = random.choice(list_nodes)
        if len(neighbors(G,x).intersection(cluster0)) != 0:
            cluster0.add(x)
            added+=1
        elif len(neighbors(G,x).intersection(cluster1)) != 0:
            cluster1.add(x)
            added+=1

    return cluster0, cluster1

def add_nodes_to_clusters(G, sample, cluster0, cluster1):
    
    cluster0_update = set()
    cluster1_update = set()

    list_nodes = [el for el in sample if el not in cluster0|cluster1 and (len(
            neighbors(G,el).intersection(cluster0)) != 0 or len(neighbors(G,el).intersection(cluster1)) != 0)]
    if len(list_nodes) > 0:
        x = random.choice(list_nodes)
        if len(neighbors(G,x).intersection(cluster0)) != 0:
            cluster0_update.add(x)
        elif len(neighbors(G,x).intersection(cluster1)) != 0:
            cluster1_update.add(x)

    return cluster0_update, cluster1_update

def parallel_two_means(G, j=2):

    n=G.number_of_nodes()

    u = random.choice(list(G.nodes()))
    v = random.choice(list(nx.non_neighbors(G, u)))
    cluster0 = {u}
    cluster1 = {v}
    added = 2

    while added < n:
        with Parallel(n_jobs = j) as parallel:
            result = parallel(delayed(add_nodes_to_clusters)(G, X, cluster0, cluster1) for X in chunks(G.nodes(),math.ceil(len(G.nodes())/j)))
            for el in result:
                cluster0.update(el[0])
                cluster1.update(el[1])
                added += len(el[0]) + len(el[1])

    return cluster0, cluster1

if __name__ == "__main__":
    
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

    # Specifica il numero di iterazioni desiderate per l'algoritmo two_means
    num_iterations = 5

    # esegui l'algoritmo in sequenza
    results = two_means(G)
    print(results)

    # Esegui l'algoritmo in parallelo
    results = parallel_two_means(G, j=9)
    print(results)

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

    results = two_means(G)  
    print(results)

    results = parallel_two_means(G, j=9)
    print(results)
