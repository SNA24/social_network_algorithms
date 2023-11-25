import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

from centrality_measures.betweenness import betweenness, parallel_betweenness
import networkx as nx
from utilities.priorityq import PriorityQueue
from utilities.parallel_algorithms import connected_components

# Clusters are computed by iteratively removing edges of largest betweenness
def girman_newman(G, threshold=None):
    graph=G.copy() # We make a copy of the graph. In this way we will modify the copy, but not the original graph

    q = nx.algorithms.community.modularity(G,connected_components(graph))

    done = False
    while not done:
        # After each edge removal we will recompute betweenness:
        # indeed, edges with lower betweenness may have increased their importance,
        # since shortest path that previously went through on deleted edges, now may be routed on this new edge;
        # similarly, edges with high betweenness may have decreased their importance,
        # since most of the shortest paths previously going through them disappeared because the graph has been disconnected.
        # However, complexity arising from recomputing betweenness at each iteration is huge.
        eb, nb = betweenness(graph)
        edge=tuple(max(eb, key=eb.get))
        # if the graph is directed, we need to check if the edge is in the graph
        if graph.has_edge(edge[0],edge[1]):
            graph.remove_edge(edge[0],edge[1])

        # We continue iteration of the algorithm as long as the newly achieved clustering
        # has performance that are not worse than the previous clustering.
        # An alternative would be to stop when performance is above a given threshold.
        newq = nx.algorithms.community.modularity(G,connected_components(graph))
        if abs(newq) <= abs(q) or (threshold is not None and abs(newq) <= threshold):
            graph.add_edge(edge[0],edge[1])
            done = True
        else:
            q = newq

    return connected_components(graph)

def heuristic_girman_newman(G, threshold=None):

    graph=G.copy() 

    q = nx.algorithms.community.modularity(G,connected_components(graph))

    # A heuristic approach in this case would be to compute betweenness only once
    # and to remove edges in decreasing order of computed betweenness.
    eb, nb = betweenness(graph)
    pq = PriorityQueue()
    for e in eb:
        pq.add(e, -eb[e])

    done = False
    while not done:
        
        edge=tuple(pq.pop())
        if graph.has_edge(edge[0],edge[1]):
            graph.remove_edge(edge[0],edge[1])

        newq = nx.algorithms.community.modularity(G,connected_components(graph))
        if abs(newq) <= abs(q) or (threshold is not None and abs(newq) <= threshold):
            graph.add_edge(edge[0],edge[1])
            done = True
        else:
            q = newq

    return connected_components(graph)

def parallel_girman_newman(G, j=2, threshold=None):

    graph=G.copy() # We make a copy of the graph. In this way we will modify the copy, but not the original graph

    q = nx.algorithms.community.modularity(G,connected_components(graph))

    done = False
    while not done:

        eb, nb = parallel_betweenness(graph, j)
        edge=tuple(max(eb, key=eb.get))
        if graph.has_edge(edge[0],edge[1]):
            graph.remove_edge(edge[0],edge[1])

        newq = nx.algorithms.community.modularity(G,connected_components(graph))
        if abs(newq) <= abs(q) or (threshold is not None and abs(newq) <= threshold):
            graph.add_edge(edge[0],edge[1])
            done = True
        else:
            q = newq

    return connected_components(graph)

def parallel_heuristic_girman_newman(G, j=2, threshold=None):
    
    graph=G.copy() 

    q = nx.algorithms.community.modularity(G,connected_components(graph))

    eb, nb = parallel_betweenness(graph, j)
    pq = PriorityQueue()
    for e in eb:
        pq.add(e, -eb[e])

    done = False
    while not done:
        
        edge=tuple(pq.pop())
        if graph.has_edge(edge[0],edge[1]):
            graph.remove_edge(edge[0],edge[1])

        newq = nx.algorithms.community.modularity(G,connected_components(graph))
        if abs(newq) <= abs(q) or (threshold is not None and abs(newq) <= threshold):
            graph.add_edge(edge[0],edge[1])
            done = True
        else:
            q = newq

    return connected_components(graph)

if __name__ == '__main__':

    print("UNDIRECTED GRAPH")

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

    print('GIRMAN NEWMAN', girman_newman(G))
    print('HEURISTIC GIRMAN NEWMAN', heuristic_girman_newman(G))
    print('PARALLEL GIRMAN NEWMAN', parallel_girman_newman(G))
    print('PARALLERL HEURISTIC GIRMAN NEWMAN', parallel_heuristic_girman_newman(G))

    print("DIRECTED GRAPH")

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

    print('GIRMAN NEWMAN', girman_newman(G))
    print('HEURISTIC GIRMAN NEWMAN', heuristic_girman_newman(G))
    print('PARALLEL GIRMAN NEWMAN', parallel_girman_newman(G))
    print('PARALLERL HEURISTIC GIRMAN NEWMAN', parallel_heuristic_girman_newman(G))
