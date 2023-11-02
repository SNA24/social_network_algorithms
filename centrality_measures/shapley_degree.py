import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed

def chunks(data, size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

# SHAPLEY DEGREE
# Compute the Shapley value for a characteristic function that extends degree centrality to coalitions.
# Specifically, the characteristic function is value(C) = |C| + |N(C)|, where N(C) is the set of nodes outside C with at least one neighbor in C.
# Even if the Shapley Value in general takes exponential time to be computed, for this particular characteristic function a polynomial time algorithm is known.
# Indeed, it has been proved that the Shapley value of node v in this case is SV[v] = 1/(1+deg(v)) + sum_{u \in N(v), u != v} 1/(1+deg(u)).
# For more information, see Michalack et al. (JAIR 2013) sec. 4.1
def shapley_degree(G, sample = None):

    if sample is None:
        sample = G.nodes()

    if not nx.is_directed(G):

        SV = {i:1/(1+G.degree(i)) for i in sample}

        for u in sample:
            for v in G[u]:
                SV[u] += 1/(1+G.degree(v))

        return SV
    
    else:

        SV_out = {i:1/(1+G.out_degree(i)) for i in sample}

        for u in sample:
            for v in G[u]:
                SV_out[u] += 1/(1+G.out_degree(v))

        return SV_out


# PARALLELIZATION
def parallel_shapley_degree(G, j=4):

    with Parallel(n_jobs=j) as parallel:
        SV = parallel(delayed(shapley_degree)(G, sample) for sample in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

    # Merge the results
    SV_final = {}
    for sv in SV:
        SV_final.update(sv)

    return SV_final


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

    print("shapley_degree")
    sd = shapley_degree(G)
    print(sorted(sd.items(), key=lambda x: x[1], reverse=True))
    print("parallel_shapley_degree")
    psd = parallel_shapley_degree(G)
    print(sorted(psd.items(), key=lambda x: x[1], reverse=True))

    # directed graph
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

    print("shapley_degree")
    sd = shapley_degree(G)
    print(sorted(sd.items(), key=lambda x: x[1], reverse=True))
    print("parallel_shapley_degree")
    psd = parallel_shapley_degree(G)
    print(sorted(psd.items(), key=lambda x: x[1], reverse=True))
