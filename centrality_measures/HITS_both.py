from joblib import Parallel, delayed
import networkx as nx
import math
from itertools import product

# HITS
# G is assumed to be a directed graph
# steps and confidence are as for pageRank
# wa and wh are the weights that we assign to the authority value and the hub value of a node, in order to evaluate its rank
# HITS
# G is assumed to be a directed graph
# steps and confidence are as for pageRank
# wa and wh are the weights that we assign to the authority value and the hub value of a node, in order to evaluate its rank
def hits(G, step=75, confidence=0, wa=0.5, wh=0.5):
    G = nx.DiGraph(G)
    time = 0
    n = nx.number_of_nodes(G)
    done = False

    hub = {i: float(1) / n for i in G.nodes()} # This contains the hub rank of each node.
    auth = {i: float(1) / n for i in G.nodes()} # This contains the authority rank of each node.

    while not done and time < step:
        time += 1

        htmp = {i:0 for i in G.nodes()} # it contains the new hub rank
        atmp = {i: 0 for i in G.nodes()} # it contains the new authority rank

        atot=0
        for i in G.nodes():
            for e in G.in_edges(i):
                # The authority level increases as better hubs are pointing to him
                atmp[i] += hub[e[0]] #the authority value of a node is the sum over all nodes pointing to him of their hubbiness value
                atot += hub[e[0]] #computes the sum of atmp[i] over all i. It serves only for normalization (each rank is done so that all values always sum to 1)

        htot=0
        for i in G.nodes():
            for e in G.out_edges(i):
                # The hubbiness level increases as it points to better authorities
                htmp[i] += auth[e[1]] #the hubbiness value of a node is the sum over all nodes at which it points of their authority value
                htot += auth[e[1]] #computes the sum of htmp[i] over all i. It serves only for normalization (each rank is done so that all values always sum to 1)

        adiff = 0
        hdiff = 0
        for i in G.nodes():
            adiff += abs(auth[i]-atmp[i] / atot)
            auth[i] = atmp[i] / atot
            hdiff += abs(hub[i] - htmp[i] / htot)
            hub[i] = htmp[i] / htot

        if adiff <= confidence and hdiff <=confidence:
            done = True

    return {i: wa*auth[i]+wh*hub[i] for i in G.nodes()}

def partition(G, n):

    if not nx.is_directed(G):
        G = G.to_directed()

    # turn nodes into a list
    nodes = list(G.nodes())

    # assign len(nodes)/n nodes to each partition
    position = {node : i//math.ceil(len(nodes)/n) for i, node in enumerate(nodes)}

    # list of lists of dictionaries with an empty set for each node
    graph = [[ dict((node, set()) for node in nodes if position[node] == i) for _ in range(n)] for i in range(n)]

    for u in G.nodes():
        for v in G[u]:
            graph[position[u]][position[v]][u].add(v)

    return graph, position

def hitsBlock(G, step=75, confidence=0, wa=0.5, wh=0.5):
    G = nx.DiGraph(G)
    time = 0
    n = nx.number_of_nodes(G)
    done = False

    hub = {i: float(1) / n for i in G.nodes()}
    auth = {i: float(1) / n for i in G.nodes()}

    while not done and time < step:
        time += 1

        htmp = {i: 0 for i in G.nodes()}
        atmp = {i: 0 for i in G.nodes()}

        atot = 0
        for i in G.nodes():
            for e in G.in_edges(i):
                atmp[i] += hub[e[0]]
                atot += hub[e[0]]

        htot = 0
        for i in G.nodes():
            for e in G.out_edges(i):
                htmp[i] += auth[e[1]]
                htot += auth[e[1]]

        adiff = 0
        hdiff = 0
        for i in G.nodes():
            if atot > 0:
                adiff += abs(auth[i] - atmp[i] / atot)
                auth[i] = atmp[i] / atot
            if htot > 0:
                hdiff += abs(hub[i] - htmp[i] / htot)
                hub[i] = htmp[i] / htot

        if adiff <= confidence and hdiff <= confidence:
            done = True

    return {i: wa*auth[i]+wh*hub[i] for i in G.nodes()}

def hitsParallel(G, step=75, confidence=0, wa=0.5, wh=0.5, n_jobs=1):
    if not math.sqrt(n_jobs).is_integer():
        print("n_jobs must be a perfect square")
        return

    n = int(math.sqrt(n_jobs))
    graph, position = partition(G, n)

    correspondence = {}
    k = 0
    for i in range(n):
        for j in range(n):
            correspondence[(i, j)] = k
            k += 1

    def getBlock(graph,i,j):
        graph = nx.DiGraph(graph[i][j]) 
        # graph.remove_nodes_from([ node for node in graph.nodes() if graph.in_degree(node) == 0 and graph.out_degree(node) == 0 ])   
        return graph

    hits_results = Parallel(n_jobs=n_jobs)(
        delayed(hitsBlock)(getBlock(graph, i, j), step, confidence, wa, wh)
        for i, j in product(range(n), repeat=2)
    )

    hub = {}
    auth = {}

    for node in G.nodes():
        part = position[node]
        total_hub = 0
        total_auth = 0
        for i in range(n):
            if node in graph[part][i]:
                total_hub += hits_results[correspondence[(part, i)]][node]/n_jobs
            if node in graph[i][part]:
                total_auth += hits_results[correspondence[(i, part)]][node]/n_jobs
        hub[node] = total_hub
        auth[node] = total_auth

    return {i: wa*auth[i]+wh*hub[i] for i in G.nodes()}

if __name__ == "__main__":
    
    n_jobs = 4

    print("Undirected Graph")
    G = nx.Graph()
    # G.add_edges_from([('x','y'),('x','z'),('x','w'),('y','w'),('w','z')])
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'F')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'G')

    print("HITS")
    print(sorted(hits(G).items(), key=lambda x: x[1], reverse=True))
    print("HITS Parallel")
    print(sorted(hitsParallel(G, n_jobs=n_jobs).items(), key=lambda x: x[1], reverse=True))