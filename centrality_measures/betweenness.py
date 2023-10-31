import networkx as nx
# Computes edge and vertex betweenness of the graph in input
# The algorithm is quite time-consuming. Indeed, its computational complexity is O(nm).
# Possible optimizations: parallelization, sampling
def betweenness(G, sample=None):

    if sample is None:
        sample = G.nodes()

    # Initialize the betweenness of each edge and vertex to 0 in sample
    edge_btw = {frozenset(e):0 for e in G.edges() if e[0] in sample and e[1] in sample}
    node_btw = {i:0 for i in G.nodes() if i in sample}

    for u in sample:
        # Compute the number of shortest paths from u to every other node
        tree = [] # It lists the nodes in the order in which they are visited
        spnum = {i:0 for i in sample} # It saves the number of shortest paths from u to i
        parents = {i:[] for i in sample} # It saves the parents of i in each of the shortest paths from u to i
        distance = {i:-1 for i in sample} # The length of the shortest path from u to i
        eflow = {frozenset(e):0 for e in G.edges() if e[0] in sample and e[1] in sample} # The number of shortest paths from u to other nodes that use the edge e
        vflow = {i:1 for i in sample} # The number of shortest paths starting from s that use the vertex i. It is initialized to 1 because the shortest path from s to i is assumed to uses that vertex once.

        #BFS
        queue = [u]
        spnum[u] = 1
        distance[u] = 0
        while len(queue) > 0:
            c = queue.pop(0)
            tree.append(c)
            for i in G[c]:
                if distance[i] == -1: # If vertex i has not been visited
                    queue.append(i)
                    distance[i] = distance[c] + 1
                if distance[i] == distance[c] + 1: # If we have found another shortest path from s to i
                    spnum[i] += spnum[c]
                    parents[i].append(c)

        #BOTTOM-UP PHASE
        while len(tree) > 0:
            c = tree.pop()
            for i in parents[c]:
                e=frozenset((c,i))
                eflow[e] += vflow[c] * (spnum[i]/spnum[c]) # The number of shortest paths using vertex c is split among the edges towards its parents proportionally to the number of shortest paths that the parents contributes
                vflow[i] += eflow[e] # Each shortest path that use an edge (i,c) where i is closest to u than c must use also vertex i
                edge_btw[e] += eflow[e] # Betweenness of an edge is the sum over all sources u of the number of shortest paths from u to other nodes using that edge
            if c != u:
                node_btw[i] += vflow[i] # Betweenness of a vertex is the sum over all s of the number of shortest paths from s to other nodes using that vertex

    return edge_btw,node_btw

if __name__ == '__main__':
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

    print("betweenness")
    print(sorted(betweenness(G)[1].items(), key=lambda x: x[1], reverse=True))