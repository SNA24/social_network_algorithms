import itertools as it
import networkx as nx

def chunks(data, size):
    idata=iter(data)
    for i in range(0, len(data), size):
        yield {k:data[k] for k in it.islice(idata, size)}

def neighbors(G, node):
    if G.is_directed():
        return set(G.predecessors(node)) | set(G.successors(node))
    else:
        return set(G.neighbors(node))
    
def connected_components(G):
    if G.is_directed():
        return list(nx.strongly_connected_components(G))
    else:
        return list(nx.connected_components(G))
    
def laplacian_matrix(G):
    if G.is_directed():
        return nx.directed_laplacian_matrix(G)
    else:
        return nx.laplacian_matrix(G)