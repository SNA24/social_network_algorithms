from joblib import Parallel, delayed
import networkx as nx
import math

#PAGE RANK
#s is the probability of selecting a neighbor. The probability of a restart is 1-s
#step is the maximum number of steps in which the process is repeated
#confidence is the maximum difference allowed in the rank between two consecutive step.
#When this difference is below or equal to confidence, we assume that computation is terminated.
def pageRank(G, s=0.85, step=75, confidence=0):
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
                tmp[v] += rank[u]/G.degree(u)

        # computes the difference between the old rank and the new rank and updates rank to contain the new rank
        # difference is computed in L1 norm.
        # Alternatives are L2 norm (Euclidean Distance) and L_infinity norm (maximum pointwise distance)
        diff = sum(abs(rank[i]-tmp[i]) for i in G.nodes())
        if diff <= confidence:
            done = True

        rank = tmp

    return rank

def partition(G, n):

    # turn nodes into a list
    nodes = list(G.nodes())

    # assign len(nodes)/n nodes to each partition
    partition = {node : i//math.ceil(len(nodes)/n) for i, node in enumerate(nodes)}

    # list of lists of dictionaries with an empty set for each node
    graph = [[ dict((node, set()) for node in nodes if partition[node] == i) for _ in range(n)] for i in range(n)]

    for u in G.nodes():
        for v in G[u]:
            graph[partition[u]][partition[v]][u].add(v)
            if not nx.is_directed(G):
                graph[partition[v]][partition[u]][v].add(u)

    return graph

    # Given this block representation of a graph, then each of the j jobs can execute the update procedure only on its block.
    # Then the page rank of a node u in the i-th partition is given by the sum of the page ranks computed by jobs that worked in blocks[*][i].
    # Note that this sum can be also be parallelized with each job executing it for different partitions.

def pageRankParallel(G, s=0.85, step=75, confidence=0, n_jobs=1):

    # partition the graph into n_jobs blocks
    graph = partition(G, n_jobs)

    def getBlock(graph,i,j):
        graph = nx.DiGraph(graph[i][j])
        return graph

    # compute the page rank for each block
    ranks = Parallel(n_jobs=n_jobs)(delayed(pageRank)(getBlock(graph,i,j), s, step, confidence) for i in range(n_jobs) for j in range(n_jobs))

    # sum the page ranks for each node
    rank = {node : sum(ranks[i][node] for i in range(n_jobs) if node in ranks[i].keys()) for node in G.nodes()}

    return rank

if __name__ == '__main__':

    G = nx.DiGraph()
    G.add_edges_from([('x','y'),('x','z'),('x','w'),('y','x'),('y','w'),('z','x'),('w','y'),('w','z')])
    print(partition(G, 2))

    print(sorted(pageRank(G).items(), key=lambda x : x[1], reverse=True))
    print(sorted(pageRankParallel(G, n_jobs=25).items(), key=lambda x : x[1], reverse=True))

    
