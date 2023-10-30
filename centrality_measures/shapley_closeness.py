import networkx as nx
from priorityq import PriorityQueue

#CENTRALITY MEASURES
#Returns the top k nodes of G according to the centrality measure "measure"
def top(G,measure,k):
    pq = PriorityQueue()
    cen=measure(G)
    for u in G.nodes():
        pq.add(u, -cen[u])  # We use negative value because PriorityQueue returns first values whose priority value is lower
    out=[]
    for i in range(k):
        out.append(pq.pop())
    return out

def function(degree):
    """
    # INPUT
    - degree: degree of a node
    # OUTPUT
    - f(degree) = 1/degree
    """
    if degree == 0:
        return 1
    return 1/degree

def shapleycloseness(G):
    """ 
    # INPUT
    - G is a networkx graph
    - f: R^+ -> R^+ is a positive decreasing function
    # OUTPUT
    - Shapley value of all the nodes in G.
    """
    SV = {i: 0 for i in G.nodes()}

    for v in G.nodes():

        distances = nx.single_source_dijkstra_path_length(G, v)
        nodes_sorted_by_distance = sorted(distances.keys(), key=lambda node: distances[node])
        w = [i for i in nodes_sorted_by_distance]
        D = [distances[i] for i in nodes_sorted_by_distance]

        sum = 0
        index = len(G.nodes())-1
        prevDistance = -1
        prevSV = -1

        while index > 0:
            if D[index] == prevDistance:
                currSV = prevSV
            else:
                currSV = function(D[index])/(1+index) - sum
            SV[w[index]] += currSV
            sum += function(D[index])/(index*(1+index))
            prevDistance = D[index]
            prevSV = currSV
            index -= 1
        
        SV[v] += function(0) - sum

    return SV

# Example
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
    # print the two centrality measures sorted by value
    print("Shapley Closeness Centrality")
    print(top(G, shapleycloseness, 7))

    