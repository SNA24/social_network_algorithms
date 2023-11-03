from joblib import Parallel, delayed
import networkx as nx
import math

# PAGE RANK
#s is the probability of selecting a neighbor. The probability of a restart is 1-s
#step is the maximum number of steps in which the process is repeated
#confidence is the maximum difference allowed in the rank between two consecutive step.
#When this difference is below or equal to confidence, we assume that computation is terminated.
def pageRank(G, s=0.85, step=75, confidence=0):

    if(len(G.nodes()) == 0):
        return {}

    time = 0
    n=nx.number_of_nodes(G)
    done = False

    # At the beginning, I choose the starting node uniformly at the random.
    # Hence, every node has the same probability of being verified at the beginning.
    rank = {i : float(1)/n for i in G.nodes()}

    while not done and time < step:
        time += 1

        # tmp contains the new rank
        # with probability 1-s, I restart the random walk. Hence, each node is visited at the next step at least with probability (1-s)*1/n
        tmp = {i : float(1-s)/n for i in G.nodes()}

        for u in G.nodes():
            for v in G[u]:
                # with probability s, I follow one of the link on the current page.
                # So, if I am on page i with probability rank[i], at the next step I would be on page j at which i links
                # with probability s*rank[i]*probability of following link (i,j) that is 1/out_degree(i)
                tmp[v] += (rank[u]/G.out_degree(u) if nx.is_directed(G) else rank[u]/G.degree(u))

        # computes the difference between the old rank and the new rank and updates rank to contain the new rank
        # difference is computed in L1 norm.
        # Alternatives are L2 norm (Euclidean Distance) and L_infinity norm (maximum pointwise distance)
        diff = sum(abs(rank[i]-tmp[i]) for i in G.nodes())
        if diff <= confidence:
            done = True

        rank = tmp

    return rank

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

    # Given this block representation of a graph, then each of the j jobs can execute the update procedure only on its block.
    # Then the page rank of a node u in the i-th partition is given by the sum of the page ranks computed by jobs that worked in blocks[*][i].
    

def pageRankParallel(G, s=0.85, step=75, confidence=0, n_jobs=1):

    # if n_jobs is not a perfect square, raise an exception
    if not math.sqrt(n_jobs).is_integer():
        print("n_jobs must be a perfect square")
        return

    n = int(math.sqrt(n_jobs))

    graph, position = partition(G, n) 

    def getBlock(graph,i,j):
        graph = nx.DiGraph(graph[i][j]) 
        # graph.remove_nodes_from([ node for node in graph.nodes() if graph.in_degree(node) == 0 and graph.out_degree(node) == 0 ])   
        return graph
    
    correspondence = {}

    k = 0
    for i in range(n):
        for j in range(n):
            correspondence[(i,j)] = k
            k += 1

    # compute the page rank for each block
    ranks = Parallel(n_jobs=n_jobs)(delayed(pageRank)(getBlock(graph,i,j), s, step, confidence) for i in range(n) for j in range(n))

    page_rank = {}

    # Then the page rank of a node u in the i-th partition is given by the sum of the page ranks computed by jobs that worked in blocks[*][i].
    for node in G.nodes():
        part = position[node]
        page_rank[node] = sum(ranks[correspondence[(i,part)]][node]/n_jobs for i in range(n) if node in ranks[correspondence[(i,part)]].keys())
    
    return page_rank

if __name__ == '__main__':

    n_jobs = 16

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

    print(sorted(pageRank(G).items(), key=lambda x : x[1], reverse=True))
    print(sorted(pageRankParallel(G, n_jobs=n_jobs).items(), key=lambda x : x[1], reverse=True))
    # print(sorted(nx.pagerank(G).items(), key=lambda x : x[1], reverse=True))

    print("Directed Graph")
    G = nx.DiGraph()
    # G.add_edges_from([('x','y'),('x','z'),('x','w'),('y','x'),('y','w'),('z','x'),('w','y'),('w','z')])
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'F')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'G')

    print(sorted(pageRank(G).items(), key=lambda x : x[1], reverse=True))
    print(sorted(pageRankParallel(G, n_jobs=n_jobs).items(), key=lambda x : x[1], reverse=True))

    
