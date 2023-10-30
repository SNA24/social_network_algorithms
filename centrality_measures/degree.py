import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed
import time

#Utility used for split a vector data in chunks of the given size.
def chunks(data, size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

#The measure associated to each node is exactly its degree
def degree(G, sample=None):

    if sample is None:
        sample = G.nodes()

    return {u:G.degree(u) for u in sample}

# 1. DIVIDE THE GRAPH IN CHUNKS
# 2. FOR EACH CHUNK, COMPUTE THE DEGREE OF EACH NODE
# 3. AGGREGATE THE RESULTS

def parallel_degree(G, j):
    cen = dict()

    # Initialize the class Parallel with the number of available process
    with Parallel(n_jobs = j) as parallel:
        # Run in parallel degree function on each processor by passing to each processor only the subset of nodes on which it works
        result = parallel(delayed(degree)(G, X) for X in chunks(G.nodes(),math.ceil(len(G.nodes())/j)))
        # Aggregates the results
        for i in result:
            cen.update(i)

    return cen

if __name__ == '__main__':
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
    first = time.time()
    print(degree(G))
    second = time.time()
    print(second - first)

    first = time.time()
    print(parallel_degree(G, 4))
    second = time.time()
    print(second - first)

