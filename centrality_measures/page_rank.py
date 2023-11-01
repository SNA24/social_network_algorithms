import networkx as nx
import math
import itertools as it
from joblib import Parallel, delayed

# PAGE RANK
#s is the probability of selecting a neighbor. The probability of a restart is 1-s
#step is the maximum number of steps in which the process is repeated
#confidence is the maximum difference allowed in the rank between two consecutive step.
#When this difference is below or equal to confidence, we assume that computation is terminated.
def pageRank(G, s=0.85, step=200, confidence=0):

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
                if nx.is_directed(G):
                    tmp[v] += rank[u]/G.out_degree(u)
                else:
                    tmp[v] += rank[u]/G.degree(u)

        # computes the difference between the old rank and the new rank and updates rank to contain the new rank
        # difference is computed in L1 norm.
        # Alternatives are L2 norm (Euclidean Distance) and L_infinity norm (maximum pointwise distance)
        diff = sum(abs(rank[i]-tmp[i]) for i in G.nodes())
        if diff <= confidence:
            done = True

        rank = tmp

    return rank

def partition(G, jobs):
    # number of subgraphs
    n = int(math.sqrt(jobs))

    graph = []
    # create the sqrt(j) partitions and for each partition create sqrt(j) subpartitions
    for i in range(n):
        graph.append([])
        for _ in range(n):
            graph[i].append(dict())

    # spread the nodes of the graph in the partitions
    positions = {}
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        # the node is assigned to the i-th partition
        partition = i % n
        # for each subpartition, create n partitions with the node as key
        for j in range(n):
            graph[partition][j][nodes[i]] = set()
            positions[nodes[i]] = (partition)

    # spread the edges of the graph in the partitions
    for u, v in G.edges():
        # get the partition of the source node
        partition = positions[u]
        # get the subpartition of the destination node
        subpartition = positions[v]
        # add the edge to the subpartition
        graph[partition][subpartition][u].add(v)

        if not nx.is_directed(G):
            # if the graph is directed, add the edge also to the partition of the destination node
            graph[subpartition][partition][v].add(u)

    return graph, positions

def print_graph(graph):
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            print("Block[{}][{}]".format(i, j))
            print(graph[i][j])

# Given this block representation of a graph, then each of the j jobs can execute the update procedure only on its block.
# Then the page rank of a node u in the i-th partition is given by the sum of the page ranks computed by jobs that worked in blocks[*][i].
# Note that this sum can be also be parallelized with each job executing it for different partitions.

def pageRank_block(graph, s=0.85, step=100, confidence=0):

    time = 0
    n = len(graph)
    done = False

    rank = {i : float(1)/n for i in graph.keys()}

    while not done and time < step:

        time += 1
        tmp = {i : float(1-s)/n for i in graph.keys()}

        for u in graph.keys():
            for v in graph[u]:
                if v in tmp.keys():
                    tmp[v] += rank[u]/len(graph[u])
                else:
                    rank[v] = float(1)/n
                    tmp[v] = rank[u]/len(graph[u])

        diff = sum(abs(rank[i]-tmp[i]) for i in tmp.keys())

        if diff <= confidence:
            done = True
        
        rank = tmp

    return rank
    
def parallel_pageRank(G, s=0.85, step=200, confidence=0, jobs=4):
    #if the number of jobs is not a perfect square, return
    if not math.sqrt(jobs).is_integer():
        return
    
    graph, positions = partition(G, jobs)
    
    with Parallel (n_jobs=jobs) as parallel:
        result = parallel(delayed(pageRank_block)(graph[i][j], s, step, confidence) for i in range(len(graph)) for j in range(len(graph[i])))

    corresponding_blocks = {}
    k = 0
    for i in range(len(graph)):
        for j in range(len(graph[i])):
            corresponding_blocks[k] = (i, j)
            k += 1

    # the page rank of a node u in the i-th partition is given by the sum of the page ranks computed by jobs that worked in blocks[*][i]
    rank = {}

    for node in G.nodes():
        rank[node] = 0
        part = positions[node]
        for k in range(len(result)):
            i,j = corresponding_blocks[k]
            if j == part:
                if node in result[k].keys():
                    rank[node] += result[k][node]

    return rank

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

    # print the graph
    print("networkx")
    nxpr = sorted(nx.pagerank(G).items(), key=lambda x: x[1], reverse=True)
    print([x[0] for x in nxpr])

    print("pageRank")
    pr = sorted(pageRank(G).items(), key=lambda x: x[1], reverse=True)
    print([x[0] for x in pr])

    print("parallel_pagerank")
    ppr = sorted(parallel_pageRank(G).items(), key=lambda x: x[1], reverse=True)
    print([x[0] for x in ppr])

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

    # print the graph
    print("networkx")
    nxpr = sorted(nx.pagerank(G).items(), key=lambda x: x[1], reverse=True)
    print([x[0] for x in nxpr])

    print("pageRank")
    pr = sorted(pageRank(G).items(), key=lambda x: x[1], reverse=True)
    print([x[0] for x in pr])

    print("parallel_pagerank")
    ppr = sorted(parallel_pageRank(G).items(), key=lambda x: x[1], reverse=True)
    print([x[0] for x in ppr])
    

