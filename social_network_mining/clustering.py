from matplotlib import pyplot as plt
from betweenness import betweenness 
from betweenness import parallel_betweenness 
from priorityq import PriorityQueue
import networkx as nx
import random
from scipy.sparse import linalg
from joblib import Parallel, delayed
import itertools as it
import math


#dividing a graph into chunks, for parallel implementation, we need to divide the nodes in chunks
def chunks(data, size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

#For direct and undirect graphs
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


#two means directed and undirected networks
def two_means(G, directed=False):
    n = G.number_of_nodes()

    if directed:
        # Per i grafi diretti, puoi utilizzare successori e predecessori
        u = random.choice(list(G.nodes()))
        v = random.choice([node for node in G.nodes() if node not in G.predecessors(u) and node not in G.successors(u) and node != u]) #controllare bene 
    else:
        # Per i grafi non diretti, puoi utilizzare i vicini
        u = random.choice(list(G.nodes()))
        v = random.choice(list(nx.non_neighbors(G, u)))

    cluster0 = {u}
    cluster1 = {v}
    added = 2

    while added < n:
        # Scegli un nodo che non è ancora in un cluster e aggiungilo al cluster più vicino
        candidate_nodes = [el for el in G.nodes() if el not in cluster0 | cluster1] 

        if directed:
            candidate_nodes = [el for el in candidate_nodes if (len(set(G.successors(el)).intersection(cluster0)) != 0 or len(set(G.predecessors(el)).intersection(cluster0)) != 0) or (len(set(G.successors(el)).intersection(cluster1)) != 0 or len(set(G.predecessors(el)).intersection(cluster1)) != 0)]
        else:
            candidate_nodes = [el for el in candidate_nodes if (len(set(G.neighbors(el)).intersection(cluster0)) != 0 or len(set(G.neighbors(el)).intersection(cluster1)) != 0)]

        if not candidate_nodes:
            break

        x = random.choice(candidate_nodes)

        if directed:
            if len(set(G.successors(x)).intersection(cluster0)) != 0 or len(set(G.predecessors(x)).intersection(cluster0)) != 0:
                cluster0.add(x)
            else:
                cluster1.add(x)
        else:
            if len(set(G.neighbors(x)).intersection(cluster0)) != 0:
                cluster0.add(x)
            else:
                cluster1.add(x)

        added += 1

    return cluster0, cluster1


#PARALLEL IMPLEMENTATION OF TWO MEANS ALGORITHM 
def two_means_v2(G,u,directed=False):
    n = G.number_of_nodes()

    if directed:
        # Per i grafi diretti, puoi utilizzare successori e predecessori
        v = random.choice([node for node in G.nodes() if node not in G.predecessors(u) and node not in G.successors(u) and node != u]) #controllare bene 
    else:
        # Per i grafi non diretti, puoi utilizzare i vicini
        v = random.choice(list(nx.non_neighbors(G, u)))

    cluster0 = {u}
    cluster1 = {v}
    added = 2

    while added < n:
        # Scegli un nodo che non è ancora in un cluster e aggiungilo al cluster più vicino
        candidate_nodes = [el for el in G.nodes() if el not in cluster0 | cluster1]

        if directed:
            candidate_nodes = [el for el in candidate_nodes if (len(set(G.successors(el)).intersection(cluster0)) != 0 or len(set(G.predecessors(el)).intersection(cluster0)) != 0) or (len(set(G.successors(el)).intersection(cluster1)) != 0 or len(set(G.predecessors(el)).intersection(cluster1)) != 0)]
        else:
            candidate_nodes = [el for el in candidate_nodes if (len(set(G.neighbors(el)).intersection(cluster0)) != 0 or len(set(G.neighbors(el)).intersection(cluster1)) != 0)]

        if not candidate_nodes:
            break

        x = random.choice(candidate_nodes)

        if directed:
            if len(set(G.successors(x)).intersection(cluster0)) != 0 or len(set(G.predecessors(x)).intersection(cluster0)) != 0:
                cluster0.add(x)
            else:
                cluster1.add(x)
        else:
            if len(set(G.neighbors(x)).intersection(cluster0)) != 0:
                cluster0.add(x)
            else:
                cluster1.add(x)

        added += 1

    return cluster0, cluster1

#Risolvere problema di avere lo stesso nodo in due cluster diversi
def parallel_two_means(G,j,directed= False):

    results = []
    '''u = []
    for i in range(j):
        u.append(random.choice(list(G.nodes())))'''
    
    #cerchiamo i cluster partendo da nodi diversi
    with Parallel (n_jobs = j) as parallel:
        starting_nodes = random.sample(list(G.nodes()), j)
        results = (parallel(delayed(two_means_v2)(G,x,directed) for x in starting_nodes))
    
    # Aggregates the results
    final_cluster0 = set()
    final_cluster1 = set()

    #COME LI AGGREGHIAMO?
    for result in results:
        c0, c1 = result
        
        final_cluster0.update(list(c0))
        final_cluster1.update(list(c1))

    return final_cluster0, final_cluster1


#Girman Newman for both directed and undirected graphs
def girman_newman(G, directed=False):

    graph=G.copy() # We make a copy of the graph. In this way we will modify the copy, but not the original graph
    
    if directed:
        connected_components = list(nx.weakly_connected_components(graph))
    else:
        connected_components = list(nx.connected_components(graph))

    q = nx.algorithms.community.modularity(G,connected_components)

    done = False
    while not done:
        # After each edge removal we will recompute betweenness:
        # indeed, edges with lower betweenness may have increased their importance,
        # since shortest path that previously went through on deleted edges, now may be routed on this new edge;
        # similarly, edges with high betweenness may have decreased their importance,
        # since most of the shortest paths previously going through them disappeared because the graph has been disconnected.
        # However, complexity arising from recomputing betweenness at each iteration is huge.
        # A heuristic approach in this case would be to compute betweenness only once
        # and to remove edges in decreasing order of computed betweenness.
        eb, nb = betweenness(graph)

        edge = tuple(max(eb, key=eb.get))

        if directed:
            if graph.has_edge(edge[0],edge[1]):
                graph.remove_edge(edge[0],edge[1]) 
            else:
                #controllare se è corretto togliere l'arco in questo modo
                graph.remove_edge(edge[1],edge[0])
        else:
            graph.remove_edge(edge[0],edge[1])
        
        # We continue iteration of the algorithm as long as the newly achieved clustering
        # has performance that are not worse than the previous clustering.
        # An alternative would be to stop when performance is above a given threshold.
        if directed:
            new_connected_components = list(nx.weakly_connected_components(graph))
        else:
            new_connected_components = list(nx.connected_components(graph))
        newq = nx.algorithms.community.modularity(G,new_connected_components)
        if abs(newq) <= abs(q):
            graph.add_edge(edge[0],edge[1])
            done = True
        else:
            q = newq

    return new_connected_components

#parallel implementation of girman newman algorithm with parallel betweenness
def parallel_girman_newman_v1 (G,j,directed=False):

    graph=G.copy() # We make a copy of the graph. In this way we will modify the copy, but not the original graph
        

    if directed:
        connected_components = list(nx.weakly_connected_components(graph))
    else:
        connected_components = list(nx.connected_components(graph))

    q = nx.algorithms.community.modularity(G,connected_components)

    done = False
    while not done:
        # After each edge removal we will recompute betweenness:
        # indeed, edges with lower betweenness may have increased their importance,
        # since shortest path that previously went through on deleted edges, now may be routed on this new edge;
        # similarly, edges with high betweenness may have decreased their importance,
        # since most of the shortest paths previously going through them disappeared because the graph has been disconnected.
        # However, complexity arising from recomputing betweenness at each iteration is huge.
        # A heuristic approach in this case would be to compute betweenness only once
        # and to remove edges in decreasing order of computed betweenness.
        eb, nb = parallel_betweenness(graph)

        edge = tuple(max(eb, key=eb.get))
            
        if directed:
            if graph.has_edge(edge[0],edge[1]):
                graph.remove_edge(edge[0],edge[1])
            else:
                graph.remove_edge(edge[1],edge[0])
        else:
            graph.remove_edge(edge[0],edge[1])

        #edge=tuple(max(eb, key=eb.get))
        
        # We continue iteration of the algorithm as long as the newly achieved clustering
        # has performance that are not worse than the previous clustering.
        # An alternative would be to stop when performance is above a given threshold.
        if directed:
            new_connected_components = list(nx.weakly_connected_components(graph))
        else:
            new_connected_components = list(nx.connected_components(graph))
        newq = nx.algorithms.community.modularity(G,new_connected_components)
        if abs(newq) <= abs(q):
            graph.add_edge(edge[0],edge[1])
            done = True
        else:
            q = newq
    
    return new_connected_components

#parallel implementation of girman newman algorithm
def girman_newman_parallel (G, j,directed=False):   
    
    results = []
    
    with Parallel (n_jobs = j) as parallel:
        pass
        #results = parallel(delayed(girman_newman_v2)(G,X,directed) for X in chunks(G.nodes(),math.ceil(len(G.nodes())/j)))
    #aggregation of results
    return results

   

   
#Spectral for directed and undirected networks
def spectral(G,directed=False):
    n = G.number_of_nodes()
    nodes = sorted(G.nodes())
    # Laplacian of a graph is a matrix, with diagonal entries being the degree of the corresponding node
    # and off-diagonal entries being -1 if an edge between the corresponding nodes exists and 0 otherwise
    if directed:
        L=nx.directed_laplacian_matrix(G, nodes).astype('f')
        #print(L)
    else:
        L=nx.laplacian_matrix(G, nodes).astype('f')
        #print(L) #To see the laplacian of G uncomment this line
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


#PARALLEL SPECTRAL IMPLEMENTATION
def spectral_v2(G,directed = False, sample = None):
    nodes = sorted(G.nodes())
    
    if sample is None:
        sample = nodes
    else:
        sample = sorted(sample)

    n = len(sample)
    # Laplacian of a graph is a matrix, with diagonal entries being the degree of the corresponding node
    # and off-diagonal entries being -1 if an edge between the corresponding nodes exists and 0 otherwise
    if directed:
        L=nx.directed_laplacian_matrix(G.subgraph(sample), sample).astype('f')
        #print(L)
    else:
        L=nx.laplacian_matrix(G.subgraph(sample), sample).astype('f')
        #print(L) #To see the laplacian of G uncomment this line
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
            c1.add(sample[i])
        else:
            c2.add(sample[i])

    # How to achieve more than two clusters? Two options:
    # (i) for each subgraph corresponding to one of the clusters, we can split this subgraph by running the spectral algorithm on it;
    # (ii) we can use further eigenvectors. For example, we can partition nodes in four clusters by using the first two eigenvectors,
    #     so that the first (second, respectively) cluster contains those nodes i such that v[i,0] and v[i,1] are both negative (both non-negative, resp.)
    #     while the third (fourth, respectively) cluster contains those nodes i such that only v[i,0] (only v[i,1], resp.) is negative.
    return (c1, c2)

#Manca l'aggregazione dei risultati
def parallel_spectral(G,j, directed=False):
    results =[]
    
    with Parallel(n_jobs=j) as parallel:
        results = parallel(delayed(spectral_v2)(G,directed,X) for X in chunks(G.nodes(),math.ceil(len(G.nodes())/j)))

    c1_final = set()
    c2_final = set()
    #aggregazione risultati
    for result in results:
        print(result)
    
    return (c1_final,c2_final)


if __name__ == '__main__':
    print("GRAFO DIRETTO")
    #GRAFO DIRETTO
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

    print("Hierarchical")
    print(hierarchical(G))
    print("Two Means")
    print(two_means(G,directed=True))
    
    print("Spectral")
    print(spectral(G,directed=True))
    
    print("Girman Newman")
    print(girman_newman(G,directed=True))

    #print("PARALLEL IMPLEMENTATION")

    '''print("Parallel Two Means")
    print(parallel_two_means(G,2,directed=True))'''

    '''print("Parallel Spectral")
    print(parallel_spectral(G,2,directed=True))'''


    '''print("Parallel Girman Newman")
    print(parallel_girman_newman_v1(G,2,directed=True))'''

    # Visualizzazione del grafo
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    labels = {edge: edge for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    plt.show()

    print("GRAFO NON DIRETTO")
    #GRAFO NON DIRETTO
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

    print("Hierarchical")
    print(hierarchical(G))
    print("Two Means")
    print(two_means(G,directed=False))
    
    print("Spectral")
    print(spectral(G,directed=False))
    
    print("Girman Newman")
    print(girman_newman(G,directed=False))

'''    print("PARALLEL IMPLEMENTATION")
    print("Parallel Two Means")
    print(parallel_two_means(G,2,directed=False))

    print("Parallel Spectral")
    print(parallel_spectral(G,2,directed=False))

    print("Parallel Girman Newman")
    print(parallel_girman_newman_v1(G,2,directed=False))'''
