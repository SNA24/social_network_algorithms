import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import math
import networkx as nx
from joblib import Parallel, delayed
from utilities.parallel_algorithms import chunks

#CENTRALITY MEASURES

def inverse(degree):
    """
    # INPUT
    - degree: degree of a node
    # OUTPUT
    - f(degree) = 1/degree
    """
    return 1/degree

def exponential_decay(degree, lambda_value=0.1):
    """
    # INPUT
    - degree: degree of a node
    - lambda_value: decay rate
    # OUTPUT
    - f(degree) = e^(-lambda_value*degree)
    """
    return math.exp(-lambda_value*degree)

def power_law(degree, alpha=0.1):
    """
    # INPUT
    - degree: degree of a node
    - alpha: power law exponent
    # OUTPUT
    - f(degree) = degree^(-alpha)
    """
    return degree**(-alpha)

def shapley_closeness(G, sample = None):
    """ 
    # INPUT
    - G is a networkx graph
    - f: R^+ -> R^+ is a positive decreasing function
    # OUTPUT
    - Shapley value of all the nodes in G.
    """
    if sample is None:
        sample = G.nodes()

    SV = {i: 0 for i in G.nodes()}

    for v in sample:

        distances = nx.single_source_dijkstra_path_length(G, v)
        
        w, D = zip(*sorted(distances.items(), key=lambda x: x[1]))

        sum = 0
        index = len(distances)-1
        prevDistance = -1
        prevSV = -1

        while index > 0:
            
            if D[index] == prevDistance:
                currSV = prevSV
            else:
                currSV = exponential_decay(D[index])/(1+index) - sum
            SV[w[index]] += currSV
            sum += exponential_decay(D[index])/(index*(1+index))
            prevDistance = D[index]
            prevSV = currSV
            index -= 1
        
        SV[v] += exponential_decay(0) - sum

    return SV

def parallel_shapley_closeness(G, j=4):

    with Parallel(n_jobs=j) as parallel:
        SV = parallel(delayed(shapley_closeness)(G, chunk) for chunk in chunks(G.nodes(), math.ceil(len(G.nodes())/j)))

    return {k: sum([sv[k] for sv in SV]) for k in SV[0]}

# Example
if __name__ == '__main__':
    print("Undirected")
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
    # print the two centrality measures sorted by value
    print("Shapley Closeness Centrality")
    print(sorted(shapley_closeness(G).items(), key=lambda x: x[1], reverse=True))
    print("Parallel Shapley Closeness Centrality")
    print(sorted(parallel_shapley_closeness(G).items(), key=lambda x: x[1], reverse=True))

    print("Directed")
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
    # print the two centrality measures sorted by value
    print("Shapley Closeness Centrality")
    print(sorted(shapley_closeness(G).items(), key=lambda x: x[1], reverse=True))
    print("Parallel Shapley Closeness Centrality")
    print(sorted(parallel_shapley_closeness(G).items(), key=lambda x: x[1], reverse=True))
    