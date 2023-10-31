from matplotlib import pyplot as plt
from priorityq import PriorityQueue
import networkx as nx
import random

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

import networkx as nx
import random

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


if __name__ == '__main__':
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
    print("CLUSTERING")
    print("Hierarchical")
    print(hierarchical(G))
    print("Two Means")
    print(two_means(G,directed=True))
    
    # Visualizzazione del grafo
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10)
    labels = {edge: edge for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')
    plt.show()