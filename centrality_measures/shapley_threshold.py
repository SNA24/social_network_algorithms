import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed

def chunks(data, size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

# SHAPLEY THRESHOLD
# Consider another extension of degree centrality.
# Specifically, we assume that to influence a node outside a coalition is necessary that at least k of its neighbors are within the coalition.
# That is, the characteristic function is value(C) = |C| + |N(C,k)|, where N(C,k) is the set of nodes outside C with at least k neighbors in C.
# Even if the Shapley Value in general takes exponential time to be computed, for this particular characteristic function a polynomial time algorithm is known.
# Indeed, it has been proved that the Shapley value of node v in this case is SV[v] = min(1,k/(1+deg(v))) + sum_{u \in N(v), u != v} max(O,(deg(u)-k+1)/(deg(u)*(1+deg(u)))
# For more information, see Michalack et al. (JAIR 2013) sec. 4.2
def shapley_threshold(G, k=2, sample = None):

    if sample is None:
        sample = G.nodes()

    if not G.is_directed():

        SV = {i:min(1,k/(1+G.degree(i))) for i in sample}

        for u in sample:
            for v in G[u]:
                weight = max(0,(G.degree(u) - k + 1)/G.degree(u))
                SV[u] += weight * 1/(1+G.degree(v))

    else: 

        SV = {i:min(1,k/(1+G.out_degree(i))) for i in sample}

        for u in sample:
            for v in G[u]:
                weight = max(0,(G.out_degree(u) - k + 1)/G.out_degree(u))
                SV[u] += weight * 1/(1+G.out_degree(v))

    return SV

# PARALLELIZATION
def parallel_shapley_threshold(G, k=2, j=4):

    with Parallel(n_jobs=j) as parallel:
        SV = parallel(delayed(shapley_threshold)(G, k, X) for X in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

    # Merge the results
    SV_final = {}
    for sv in SV:
        SV_final.update(sv)

    return SV_final

if __name__ == '__main__':

    print("undirected graph")
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
    print("shapley_threshold")
    st = shapley_threshold(G, k=2)
    print(sorted(st.items(), key=lambda x: x[1], reverse=True))
    print("parallel_shapley_threshold")
    pst = parallel_shapley_threshold(G, k=2)
    print(sorted(pst.items(), key=lambda x: x[1], reverse=True))

    print("directed graph")
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
    print("shapley_threshold")
    st = shapley_threshold(G, k=2)
    print(sorted(st.items(), key=lambda x: x[1], reverse=True))
    print("parallel_shapley_threshold")
    pst = parallel_shapley_threshold(G, k=2)
    print(sorted(pst.items(), key=lambda x: x[1], reverse=True))
