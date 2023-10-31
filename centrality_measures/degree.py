import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed

#Utility used for split a vector data in chunks of the given size.
def chunks(data, size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

#The measure associated to each node is exactly its degree
def degree(G, sample=None):

    if sample is None:
        sample = G.nodes()

    if G.is_directed():
        return {u:G.in_degree(u) for u in sample}, {u:G.out_degree(u) for u in sample}

    return {u:G.degree(u) for u in sample}

# 1. DIVIDE THE GRAPH IN CHUNKS
# 2. FOR EACH CHUNK, COMPUTE THE DEGREE OF EACH NODE
# 3. AGGREGATE THE RESULTS

def parallel_degree(G, j):

    if not G.is_directed():

        cen = {}

        with Parallel(n_jobs=j) as parallel:
            results = parallel(delayed(degree)(G, sample) for sample in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

            for i in results:
                cen.update(i)

        return cen

    else: 

        in_cen = {}
        out_cen = {}

        with Parallel(n_jobs=j) as parallel:
            results = parallel(delayed(degree)(G, sample) for sample in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

            for i in results:
                in_cen.update(i[0])
                out_cen.update(i[1])

        return in_cen, out_cen

if __name__ == '__main__':
    
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

    deg_1 = degree(G)
    deg_2 = parallel_degree(G, 7)
    deg_3 = nx.degree_centrality(G)

    # print only the keys of all three centrality measures sorting by value
    print(sorted(deg_1.items(), key=lambda x: x[1]))
    print(sorted(deg_2.items(), key=lambda x: x[1]))
    print(sorted(deg_3.items(), key=lambda x: x[1]))

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

    in_deg_1, out_deg_1 = degree(G)
    in_deg_2, out_deg_2 = parallel_degree(G, 7)
    in_deg_3 = nx.in_degree_centrality(G)
    out_deg_3 = nx.out_degree_centrality(G)

    # print all three centrality measures sorting by value
    print(sorted(in_deg_1.items(), key=lambda x: x[1]))
    print(sorted(out_deg_1.items(), key=lambda x: x[1]))
    print(sorted(in_deg_2.items(), key=lambda x: x[1]))
    print(sorted(out_deg_2.items(), key=lambda x: x[1]))
    print(sorted(in_deg_3.items(), key=lambda x: x[1]))
    print(sorted(out_deg_3.items(), key=lambda x: x[1]))



