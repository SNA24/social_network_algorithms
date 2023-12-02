import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import networkx as nx
import math
from joblib import Parallel, delayed
from utilities.parallel_algorithms import chunks

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

        return SV

    else: 

        SV_in = {i:min(1,k/(1+G.in_degree(i))) for i in sample}
        SV_out = {i:min(1,k/(1+G.out_degree(i))) for i in sample}

        for u in sample:
            for v in G.successors(u):
                weight = max(0,(G.out_degree(u) - k + 1)/G.out_degree(u))
                SV_out[u] += weight * 1/(1+G.out_degree(v))
            for v in G.predecessors(u):
                weight = max(0,(G.in_degree(u) - k + 1)/G.in_degree(u))
                SV_in[u] += weight * 1/(1+G.in_degree(v))

        return SV_in, SV_out

def parallel_shapley_threshold(G, j=4, k=2):

    if not G.is_directed():

        with Parallel(n_jobs=j) as parallel:
            SV = parallel(delayed(shapley_threshold)(G, k, sample) for sample in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

        # Merge the results
        SV_final = {}
        for sv in SV:
            SV_final.update(sv)

        return SV_final
    
    else:

        with Parallel(n_jobs=j) as parallel:
            SV_in, SV_out = zip(*parallel(delayed(shapley_threshold)(G, k, sample) for sample in chunks(G.nodes(), math.ceil(len(G.nodes())/j))))

        # Merge the results
        SV_in_final = {}
        for sv in SV_in:
            SV_in_final.update(sv)

        SV_out_final = {}
        for sv in SV_out:
            SV_out_final.update(sv)

        return SV_in_final, SV_out_final

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
    print(sorted(shapley_threshold(G, k=2).items(), key=lambda x: x[1], reverse=True))
    print("parallel_shapley_threshold")
    print(sorted(parallel_shapley_threshold(G, k=2).items(), key=lambda x: x[1], reverse=True))

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
    print("in")
    st_in = shapley_threshold(G, k=2)[0]
    print(sorted(st_in.items(), key=lambda x: x[1], reverse=True))
    print("out")
    st_out = shapley_threshold(G, k=2)[1]
    print(sorted(st_out.items(), key=lambda x: x[1], reverse=True))

    print("parallel_shapley_threshold")
    print("in")
    pst_in = parallel_shapley_threshold(G, k=2)[0]
    print(sorted(pst_in.items(), key=lambda x: x[1], reverse=True))
    print("out")
    pst_out = parallel_shapley_threshold(G, k=2)[1]
    print(sorted(pst_out.items(), key=lambda x: x[1], reverse=True))

