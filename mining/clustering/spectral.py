import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

import networkx as nx
from scipy.sparse import linalg
from joblib import Parallel, delayed
from utilities.parallel_algorithms import laplacian_matrix
import math

#Spectral clustering algorithm
def spectral(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    # Laplacian of a graph is a matrix, with diagonal entries being the degree of the corresponding node
    # and off-diagonal entries being -1 if an edge between the corresponding nodes exists and 0 otherwise
    L=laplacian_matrix(G).astype('f')
    # print(L) #To see the laplacian of G uncomment this line
    # The following command computes eigenvalues and eigenvectors of the Laplacian matrix.
    # Recall that these are scalar numbers w_1, ..., w_k and vectors v_1, ..., v_k such that Lv_i=w_iv_i.
    # The first output is the array of eigenvalues in increasing order.
    # The second output contains the matrix of eigenvectors:
    # specifically, the eigenvector of the k-th eigenvalue is given by the k-th column of v
    w, v = linalg.eigsh(L,n-1)
    # print(w) #Print the list of eigenvalues
    # print(v) #Print the matrix of eigenvectors
    # print(v[:,0]) #Print the eigenvector corresponding to the first returned eigenvalue

    # Partition in clusters based on the corresponding eigenvector value being positive or negative
    # This is known to return (an approximation of) the sparset cut of the graph
    # That is, the cut with each of the clusters having many edges, and with few edge among clusters
    # Note that this is not the minimum cut (that only requires few edge among clusters,
    # but it does not require many edge within clusters)
    c1 = set()
    c2 = set()

    for i in range(n):
        if v[i,0] < 0:
            c1.add(nodes[i])
        else:
            c2.add(nodes[i])

    return (c1, c2)

def partition_into_clusters(nodes, v):

    c1 = set()
    c2 = set()

    for i in range(len(nodes)):
        if v[i,0] < 0:
            c1.add(nodes[i])
        else:
            c2.add(nodes[i])

    return (c1, c2)

def split_list_and_eigenvector(nodes, j, v):
    n = len(nodes)
    step = n // j
    for i in range(0, n, step):
        # check if we are at the end of the list
        if i+step > n:
            yield nodes[i:], v[i:]
        else:
            yield nodes[i:i+step], v[i:i+step]

def split_list(nodes, j):
    n = len(nodes)
    step = n // j
    for i in range(0, n, step):
        # check if we are at the end of the list
        if i+step > n:
            yield i, n
        else:
            yield i, i+step

def spectral_parallel(G, j=2):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())

    L=laplacian_matrix(G).astype('f')
    w, v = linalg.eigsh(L,n-1)

    # if j is greater than the number of nodes, set j to the number of nodes
    if j > n:
        j = n

    with Parallel(n_jobs=j) as parallel:
        results = parallel(delayed(partition_into_clusters)(nodes, v) for nodes, v in split_list_and_eigenvector(nodes, j, v) for j in range(n))
        c1 = set()
        c2 = set()
        for c1_, c2_ in results:
            c1.update(c1_)
            c2.update(c2_)

    return (c1, c2)

def spectral_multi_cluster(G):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    L = laplacian_matrix(G).astype('f')
    w, v = linalg.eigsh(L, n-1)

    c1 = set()
    c2 = set()

    for i in range(n):
        if v[i,0] < 0:
            c1.add(nodes[i])
        else:
            c2.add(nodes[i])

    if len(c1) > 1:
        c1 = spectral(nx.subgraph(G, c1))
    if len(c2) > 1:
        c2 = spectral(nx.subgraph(G, c2))

    # merge the tuples of clusters
    return c1 + c2

def spectral_multi_cluster_parallel(G, j=2):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())

    L=laplacian_matrix(G).astype('f')
    w, v = linalg.eigsh(L,n-1)

    # if j is greater than the number of nodes, set j to the number of nodes
    if j > n:
        j = n

    with Parallel(n_jobs=j) as parallel:
        results = parallel(delayed(partition_into_clusters)(nodes, v) for nodes, v in split_list_and_eigenvector(nodes, j, v) for j in range(n))
        c1 = set()
        c2 = set()
        for c1_, c2_ in results:
            c1.update(c1_)
            c2.update(c2_)

    if len(c1) > 1:
        c1 = spectral_parallel(nx.subgraph(G, c1))
    if len(c2) > 1:
        c2 = spectral_parallel(nx.subgraph(G, c2))

    return c1 + c2

def compute_cluster_index(signs):
    # Convert binary representation to decimal representation
    cluster_index = 0
    for i in range(len(signs)):
        cluster_index += signs[i] * (2 ** i)
    return cluster_index

def spectral_multi_cluster_v2(G, num_clusters=4):
    n = G.number_of_nodes()

    # check if num_clusters is a perfect square
    if math.sqrt(num_clusters) % 1 != 0:
        # set num_clusters to the next perfect square
        num_clusters = int(math.ceil(math.sqrt(num_clusters)) ** 2)
    
    nodes = sorted(G.nodes())
    L = laplacian_matrix(G).astype('f')
    w, v = linalg.eigsh(L, n-1)

    num_eigenvectors = int(math.sqrt(num_clusters))

    clusters = [set() for _ in range(num_clusters)]
    for i in range(n):
        signs = [1 if v[i, j] >= 0 else 0 for j in range(num_eigenvectors)]
        cluster_index = compute_cluster_index(signs)
        clusters[cluster_index].add(nodes[i])

    return [cluster for cluster in clusters if len(cluster) > 0]

def populate_clusters(v, num_eigenvectors, nodes, clusters, start_i, stop_i):
    for i in range(start_i, stop_i):
        signs = [1 if v[i, j] >= 0 else 0 for j in range(num_eigenvectors)]
        cluster_index = compute_cluster_index(signs)
        clusters[cluster_index].add(nodes[i])
    return clusters

def spectral_multi_cluster_v2_parallel(G, j=2, num_clusters=4):
    n = G.number_of_nodes()
    
    # check if num_clusters is a perfect square
    if math.sqrt(num_clusters) % 1 != 0:
        # set num_clusters to the next perfect square
        num_clusters = int(math.ceil(math.sqrt(num_clusters)) ** 2)
    
    nodes = sorted(G.nodes())
    L = laplacian_matrix(G).astype('f')
    w, v = linalg.eigsh(L, n-1)

    num_eigenvectors = int(math.sqrt(num_clusters))

    clusters = [set() for _ in range(num_clusters)]

    with Parallel(n_jobs=j) as parallel:
        clus = parallel(delayed(populate_clusters)(v, num_eigenvectors, nodes, clusters, start_i, stop_i) for start_i, stop_i in split_list(nodes, j))
        for cluster in clus:
            for i in range(len(cluster)):
                clusters[i].update(cluster[i])

    return [cluster for cluster in clusters if len(cluster) > 0]
    
if __name__ == '__main__':

    jobs = 6

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

    print("SPECTRAL", spectral(G))
    print("SPECTRAL PARALLEL", spectral_parallel(G, jobs))
    print("SPECTRAL MULTI CLUSTER", spectral_multi_cluster(G))
    print("SPECTRAL MULTI CLUSTER PARALLEL", spectral_multi_cluster_parallel(G, jobs))
    print("SPECTRAL MULTI CLUSTER V2", spectral_multi_cluster_v2(G))
    print("SPECTRAL MULTI CLUSTER V2 PARALLEL", spectral_multi_cluster_v2_parallel(G, jobs))

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

    print("SPECTRAL", spectral(G))
    print("SPECTRAL PARALLEL", spectral_parallel(G, jobs))
    print("SPECTRAL MULTI CLUSTER", spectral_multi_cluster(G))
    print("SPECTRAL MULTI CLUSTER PARALLEL", spectral_multi_cluster_parallel(G, jobs))
    print("SPECTRAL MULTI CLUSTER V2", spectral_multi_cluster_v2(G))
    print("SPECTRAL MULTI CLUSTER V2 PARALLEL", spectral_multi_cluster_v2_parallel(G, jobs))