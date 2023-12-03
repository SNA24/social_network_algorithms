import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from joblib import Parallel, delayed
import networkx as nx
import math

# HITS
# G is assumed to be a directed graph
# steps and confidence are as for pageRank
# wa and wh are the weights that we assign to the authority value and the hub value of a node, in order to evaluate its rank
def hits(G, step=75, confidence=0, wa=0.5, wh=0.5):
    
    if not G.is_directed():
        G = G.to_directed()

    time = 0
    n = nx.number_of_nodes(G)
    done = False

    hub = {i: float(1) / n for i in G.nodes()} # This contains the hub rank of each node.
    auth = {i: float(1) / n for i in G.nodes()} # This contains the authority rank of each node.

    while not done and time < step:
        time += 1

        htmp = {i: 0 for i in G.nodes()} # it contains the new hub rank
        atmp = {i: 0 for i in G.nodes()} # it contains the new authority rank

        atot=0
        for i in G.nodes():
            for u in G.predecessors(i):
                # The authority level increases as better hubs are pointing to him
                atmp[i] += hub[u] #the authority value of a node is the sum over all nodes pointing to him of their hubbiness value
                atot += hub[u] #computes the sum of atmp[i] over all i. It serves only for normalization (each rank is done so that all values always sum to 1)

        htot=0
        for i in G.nodes():
            for u in G.successors(i):
                # The hubbiness level increases as it points to better authorities
                htmp[i] += auth[u] #the hubbiness value of a node is the sum over all nodes at which it points of their authority value
                htot += auth[u] #computes the sum of htmp[i] over all i. It serves only for normalization (each rank is done so that all values always sum to 1)

        adiff = 0
        hdiff = 0
        for i in G.nodes():
            adiff += abs(auth[i]-atmp[i] / atot)
            auth[i] = atmp[i] / atot
            hdiff += abs(hub[i] - htmp[i] / htot)
            hub[i] = htmp[i] / htot

        if adiff <= confidence and hdiff <=confidence:
            done = True

    return {i: wa*auth[i]+wh*hub[i] for i in G.nodes()}, hub, auth

def hits_both(G, step=75, confidence=0, wa=0.5, wh=0.5):
    return hits(G, step, confidence, wa, wh)[0]

def hits_hubbiness(G, step=75, confidence=0):
    return hits(G, step, confidence)[1]
    
def hits_authority(G, step=75, confidence=0):
    return hits(G, step, confidence)[2]

def partition(G, n):
    nodes = list(G.nodes())
    node_to_block = {node : i for i in range(n) for node in nodes if i*len(nodes)/n <= nodes.index(node) < (i+1)*len(nodes)/n}
    out_graph = [ [ { node : set() for node in nodes if node_to_block[node] == i} for _ in range(n) ] for i in range(n) ]
    in_graph = [ [ { node : set() for node in nodes if node_to_block[node] == i} for _ in range(n) ] for i in range(n) ]

    for node in nodes:

        for neighbor in G.predecessors(node):
            in_graph[node_to_block[node]][node_to_block[neighbor]][node].add(neighbor)
            
        for neighbor in G.successors(node):
            out_graph[node_to_block[node]][node_to_block[neighbor]][node].add(neighbor)

    return in_graph, out_graph, node_to_block

def update(in_block, out_block, auth, hub):

    htmp = {i: 0 for i in out_block.keys()} # it contains the new hub rank
    atmp = {i: 0 for i in in_block.keys()} # it contains the new authority rank

    atot=0
    for i in in_block.keys():
        for u in in_block[i]:
            # The authority level increases as better hubs are pointing to him
            atmp[i] += hub[u]
            atot += hub[u]

    htot=0
    for i in out_block.keys():
        for u in out_block[i]:
            htmp[i] += auth[u] #the hubbiness value of a node is the sum over all nodes at which it points of their authority value
            htot += auth[u]

    return htmp, atmp, htot, atot

def parallel_hits(G, n_jobs = 4, step=75, confidence=0, wa=0.5, wh=0.5):
    
    if not G.is_directed():
        G = G.to_directed()

    if not math.sqrt(n_jobs).is_integer():
        n_jobs = int(math.sqrt(n_jobs))**2

    num_partitions = int(math.sqrt(n_jobs))

    in_graph, out_graph, node_to_block = partition(G, num_partitions)

    time = 0
    n = nx.number_of_nodes(G)
    done = False

    hub = {i: float(1) / n for i in G.nodes()} # This contains the hub rank of each node.
    auth = {i: float(1) / n for i in G.nodes()} # This contains the authority rank of each node.

    jobs = {(i,j): i*num_partitions+j for i in range(num_partitions) for j in range(num_partitions)}

    while not done and time < step:

        time += 1

        with Parallel(n_jobs=n_jobs) as parallel:
            htmps, atmps, htots, atots = zip(*parallel(delayed(update)(in_graph[i][j], out_graph[i][j], auth, hub) for i in range(num_partitions) for j in range(num_partitions)))
        
        htmp = {i: 0 for i in G.nodes()} # it contains the new hub rank
        atmp = {i: 0 for i in G.nodes()}
        atot = 0
        htot = 0

        for i in range(num_partitions):
            for j in range(num_partitions):
                for node in in_graph[i][j].keys():
                    htmp[node] += htmps[jobs[(i,j)]][node]
                for node in out_graph[i][j].keys():
                    atmp[node] += atmps[jobs[(i,j)]][node]
                atot += atots[jobs[(i,j)]]
                htot += htots[jobs[(i,j)]]

        adiff = 0
        hdiff = 0
        for i in G.nodes():
            adiff += abs(auth[i] - atmp[i] / atot)
            auth[i] = atmp[i] / atot
            hdiff += abs(hub[i] - htmp[i] / htot)
            hub[i] = htmp[i] / htot

        if adiff <= confidence and hdiff <=confidence:
            done = True

    return {i: wa*auth[i]+wh*hub[i] for i in G.nodes()}, hub, auth

def parallel_hits_both(G, n_jobs = 4, step=75, confidence=0):
    return parallel_hits(G, n_jobs, step, confidence)[0]

def parallel_hits_hubbiness(G, n_jobs = 4, step=75, confidence=0):
    return parallel_hits(G, n_jobs, step, confidence)[1]
     
def parallel_hits_authority(G, n_jobs = 4, step=75, confidence=0):  
    return parallel_hits(G, n_jobs, step, confidence)[2]

if __name__ == "__main__":
    
    n_jobs = 9

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
    print(sorted(hits_both(G).items(), key=lambda x: x[1], reverse=True))
    print("HITS Parallel")
    print(sorted(parallel_hits_both(G, n_jobs=n_jobs).items(), key=lambda x: x[1], reverse=True))
    print("HITS Hubbiness")
    print(sorted(hits_hubbiness(G).items(), key=lambda x: x[1], reverse=True))
    print("HITS Hubbiness Parallel")
    print(sorted(parallel_hits_hubbiness(G, n_jobs=n_jobs).items(), key=lambda x: x[1], reverse=True))
    print("HITS Authority")
    print(sorted(hits_authority(G).items(), key=lambda x: x[1], reverse=True))
    print("HITS Authority Parallel")
    print(sorted(parallel_hits_authority(G, n_jobs=n_jobs).items(), key=lambda x: x[1], reverse=True))

    print("Directed Graph")
    G = nx.DiGraph()
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
    print(sorted(hits_both(G).items(), key=lambda x: x[1], reverse=True))
    print("HITS Parallel")
    print(sorted(parallel_hits_both(G, n_jobs=n_jobs).items(), key=lambda x: x[1], reverse=True))
    print("HITS Hubbiness")
    print(sorted(hits_hubbiness(G).items(), key=lambda x: x[1], reverse=True))
    print("HITS Hubbiness Parallel")
    print(sorted(parallel_hits_hubbiness(G, n_jobs=n_jobs).items(), key=lambda x: x[1], reverse=True))
    print("HITS Authority")
    print(sorted(hits_authority(G).items(), key=lambda x: x[1], reverse=True))
    print("HITS Authority Parallel")
    print(sorted(parallel_hits_authority(G, n_jobs=n_jobs).items(), key=lambda x: x[1], reverse=True))
