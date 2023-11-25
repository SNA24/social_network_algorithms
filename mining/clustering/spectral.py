import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

import networkx as nx
from scipy.sparse import linalg
from joblib import Parallel, delayed
from utilities.parallel_algorithms import laplacian_matrix

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

    # How to achieve more than two clusters? Two options:
    # (i) for each subgraph corresponding to one of the clusters, we can split this subgraph by running the spectral algorithm on it;
    # (ii) we can use further eigenvectors. For example, we can partition nodes in four clusters by using the first two eigenvectors,
    #     so that the first (second, respectively) cluster contains those nodes i such that v[i,0] and v[i,1] are both negative (both non-negative, resp.)
    #     while the third (fourth, respectively) cluster contains those nodes i such that only v[i,0] (only v[i,1], resp.) is negative.
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
        yield nodes[i:i+step], v[i:i+step]

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

if __name__ == '__main__':

    jobs = 16

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