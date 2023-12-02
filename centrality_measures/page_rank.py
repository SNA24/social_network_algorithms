import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from joblib import Parallel, delayed
import networkx as nx
import math
from utilities.parallel_algorithms import neighbors

# PAGE RANK
#s is the probability of selecting a neighbor. The probability of a restart is 1-s
#step is the maximum number of steps in which the process is repeated
#confidence is the maximum difference allowed in the rank between two consecutive step.
#When this difference is below or equal to confidence, we assume that computation is terminated.
def page_rank(G, s=0.85, step=75, confidence=0):

    if not G.is_directed():
        G = G.to_directed()

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
            for v in neighbors(G, u):
                # with probability s, I follow one of the link on the current page.
                # So, if I am on page i with probability rank[i], at the next step I would be on page j at which i links
                # with probability s*rank[i]*probability of following link (i,j) that is 1/out_degree(i)
                tmp[v] += rank[u]/G.out_degree(u)
                
        # computes the difference between the old rank and the new rank and updates rank to contain the new rank
        # difference is computed in L1 norm.
        # Alternatives are L2 norm (Euclidean Distance) and L_infinity norm (maximum pointwise distance)
        diff = sum(abs(rank[i]-tmp[i]) for i in G.nodes())
        if diff <= confidence:
            done = True

        rank = tmp

    return rank
    # How to parallelize Page Rank?
    # For simplicity let us assume that the number j of jobs is a perfect square (4, 9, 16, ...)
    # The idea is to split the graph in as many blocks as the number of jobs.
    # To this aim, we partition the nodes of the graph in sqrt(j) sets.
    # Then, block[0][0] will contain all edges from nodes in the first partition to nodes in the first partition,
    # block[0][1] will contain all edges from nodes in the first partition to nodes in the second partition, and so on.
    # For example, if j = 4, the directed graph G defined by the following directed edges:
    # (x,y), (x,z), (x,w), (y,x), (y,w), (z,x), (w,y), (w,z)
    # should be represented as follows:
    # graph = []
    # graph.append([]) #The first row of blocks
    # graph.append([]) #The second row of blocks

    ##The block[0][0]
    # graph[0].append(dict())
    # graph[0][0]['x']= {'y'}
    # graph[0][0]['y']= {'x'}

    ##The block[0][1]
    # graph[0].append(dict())
    # graph[0][1]['x']= {'z','w'}
    # graph[0][1]['y']= {'w'}

    ##The block[1][0]
    # graph[1].append(dict())
    # graph[1][0]['z']= {'x'}
    # graph[1][0]['w']= {'y'}

    ##The block[1][1]
    # graph[1].append(dict())
    # graph[1][1]['z']= {}
    # graph[1][1]['w']= {'z'}

    # Given this block representation of a graph, then each of the j jobs can execute the update procedure only on its block.
    # Then the page rank of a node u in the i-th partition is given by the sum of the page ranks computed by jobs that worked in blocks[*][i].
    # Note that this sum can be also be parallelized with each job executing it for different partitions.

def partition(G, n):

    nodes = list(G.nodes())
    node_to_block = {node : i for i in range(n) for node in nodes if i*len(nodes)/n <= nodes.index(node) < (i+1)*len(nodes)/n}
    graph = [ [ {(node, G.out_degree(node)): set() for node in nodes if node_to_block[node] == i} for _ in range(n) ] for i in range(n)]

    for node in nodes:
        for neighbor in neighbors(G, node):
            graph[node_to_block[node]][node_to_block[neighbor]][(node, G.out_degree(node))].add((neighbor, G.out_degree(neighbor)))

    return graph, node_to_block
    
def update(block, s, n, rank):

    # tmp contains the new rank
    # with probability 1-s, I restart the random walk. Hence, each node is visited at the next step at least with probability (1-s)*1/n
    tmp = {i[0] : float(1-s)/n for i in block.keys()}

    for u in block.keys():
        for v in block[u]:
            # with probability s, I follow one of the link on the current page.
            # So, if I am on page i with probability rank[i], at the next step I would be on page j at which i links
            # with probability s*rank[i]*probability of following link (i,j) that is 1/out_degree(i)
            if v[0] in tmp.keys():
                tmp[v[0]] += rank[u[0]]/u[1]
            else:
                tmp[v[0]] = rank[u[0]]/u[1]

    return tmp

def sum_page_ranks(tmps, jobs, node_to_block, num_partitions, node):
    return sum(tmps[jobs[(i, node_to_block[node])]][node] if node in tmps[jobs[(i, node_to_block[node])]].keys() else 0 for i in range(num_partitions))

def parallel_page_rank(G, n_jobs=4, s=0.85, step=75, confidence=0):

    if not G.is_directed():
        G = G.to_directed()
    
    # n_jobs must be a perfect square
    if not math.sqrt(n_jobs).is_integer():
        # approximates the number of jobs to the nearest perfect square
        n_jobs = int(math.sqrt(n_jobs))**2
    
    num_partitions = int(math.sqrt(n_jobs))
    
    graph, node_to_block = partition(G, num_partitions)

    time = 0
    n=nx.number_of_nodes(G)
    done = False

    # At the beginning, I choose the starting node uniformly at the random.
    # Hence, every node has the same probability of being verified at the beginning.
    rank = {i : float(1)/n for i in G.nodes()}

    jobs = {(i,j): i*num_partitions+j for i in range(num_partitions) for j in range(num_partitions)}

    while not done and time < step:
        time += 1

        # tmp contains the new rank
        # with probability 1-s, I restart the random walk. Hence, each node is visited at the next step at least with probability (1-s)*1/n
        tmp = {i : 0 for i in G.nodes()}

        # Given this block representation of a graph, then each of the j jobs can execute the update procedure only on its block.
        # Then the page rank of a node u in the i-th partition is given by the sum of the page ranks computed by jobs that worked in blocks[*][i].
        # Note that this sum can be also be parallelized with each job executing it for different partitions.

        # Parallelize the update procedure
        with Parallel(n_jobs=n_jobs) as parallel:
            tmps = parallel(delayed(update)(graph[i][j], s, n, rank) for i in range(num_partitions) for j in range(num_partitions))
        
        with Parallel(n_jobs=n_jobs) as parallel:
            aggregated_ranks = parallel(delayed(sum_page_ranks)(tmps, jobs, node_to_block, num_partitions, node) for node in G.nodes())

        for idx, node in enumerate(G.nodes()):
            tmp[node] = aggregated_ranks[idx]

        # computes the difference between the old rank and the new rank and updates rank to contain the new rank
        # difference is computed in L1 norm.
        # Alternatives are L2 norm (Euclidean Distance) and L_infinity norm (maximum pointwise distance)
        diff = sum(abs(rank[i]-tmp[i]) for i in G.nodes())
        if diff <= confidence:
            done = True

        rank = tmp

    return rank

if __name__ == '__main__':

    n_jobs = 4

    print("Undirected Graph")
    G = nx.Graph()
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

    print("Page Rank", sorted(page_rank(G).items(), key=lambda x : x[1], reverse=True))
    print("Parallel Page Rank", sorted(parallel_page_rank(G, n_jobs=n_jobs).items(), key=lambda x : x[1], reverse=True))

    print("Directed Graph")
    G = nx.DiGraph()
    # # G.add_edges_from([('x','y'),('x','z'),('x','w'),('y','x'),('y','w'),('z','x'),('w','y'),('w','z')])
    G.add_edge('A', 'B')
    G.add_edge('A', 'C')
    G.add_edge('B', 'C')
    G.add_edge('B', 'D')
    G.add_edge('D', 'E')
    G.add_edge('D', 'F')
    G.add_edge('D', 'G')
    G.add_edge('E', 'F')
    G.add_edge('F', 'G')

    print("Page Rank", sorted(page_rank(G).items(), key=lambda x : x[1], reverse=True))
    print("Parallel Page Rank", sorted(parallel_page_rank(G, n_jobs=n_jobs).items(), key=lambda x : x[1], reverse=True))

    
