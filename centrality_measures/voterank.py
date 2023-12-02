import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from joblib import Parallel, delayed
import networkx as nx
import math
from utilities.parallel_algorithms import chunks, split_list, neighbors, degree

#VOTERANK
# The following is an extension of degree centrality, that tries to avoid to choose nodes that are too close each other
# The idea is to simulate a voting procedure on the network, in which each node votes for each neighbor on the network
# The node that takes more votes is elected and does not participate in the next elections (hence, her neighbors do not take her vote)
# Moreover, we decrease the weight of the votes issued by the neighbors of the elected node (hence, not only the elected node does not vote, but the vote of its neighbors are less influent)
# By convention, the weight of these nodes is decreased of an amount f = 1 / average degree of the network
# For more information see Zhang et al. (Nature 2016).
def voterank(G):
    G = nx.DiGraph(G)
    rank = dict()

    n = G.number_of_nodes()
    avg_deg = sum(degree(G,v) for v in G.nodes())/n # It computes the average degree of the network
    f = 1/avg_deg # It sets the decrement of weight in the network

    ability = {i : 1 for i in G.nodes()} # Initially the vote of each node weights 1

    for i in range(n):

        score = {i : 0 for i in G.nodes() if i not in rank.keys()}

        for u in score.keys():
            for v in neighbors(G,u):
                if v not in rank.keys():
                    # the score of a node is the sum of the vote weights of the neigbors of this node that have not been elected yet
                    score[v] += ability[u]

        # computes the elected vertex
        vertex = max(score, key=score.get)
        # assigns the highest possible rank to this vertex
        rank[vertex] = n-i
        # reduces to 0 the vote weight of this vertex
        ability[vertex] = 0
        # reduces by f the vote weight of her neighbors
        for e in G.in_edges(vertex):
            ability[e[0]] = max(0,ability[e[0]]-f)

    return rank

def vote (G, score, elected, ability):

    new_score = {i : 0 for i in G.nodes() if i not in elected}

    for u in score.keys():
        for v in neighbors(G,u):
            if v not in elected:
                # the score of a node is the sum of the vote weights of the neigbors of this node that have not been elected yet
                new_score[v] += ability[u]

    return new_score

def update(neighbors, ability, f):

    new_ability = {}

    for e in neighbors:
        new_ability[e[0]] = max(0,ability[e[0]]-f)

    return new_ability

# PARLLELIZATION
def parallel_voterank(G, n_jobs=2):

    G = nx.DiGraph(G)
    rank = dict()

    n = G.number_of_nodes()
    avg_deg = sum(degree(G,v) for v in G.nodes()) / n
    f = 1 / avg_deg
    ability = {i: 1 for i in G.nodes()}

    for i in range(n):

        scores = {i : 0 for i in G.nodes() if i not in rank.keys()}

        with Parallel(n_jobs=n_jobs) as parallel:
            results = parallel(delayed(vote)(G, score, rank.keys(), ability) for score in chunks(scores, math.ceil(len(scores)/n_jobs)))

            for j in results:
                for k in j:
                    scores[k] += j[k]

        # Find the elected node
        vertex = max(scores, key=scores.get)
        rank[vertex] = n - i
        ability[vertex] = 0

        # Update ability in parallel
        neighbors_list = [e for e in G.in_edges(vertex)]
        # create a sublist of neighbors for each job 
        neighbors_list = list(split_list(neighbors_list, math.ceil(len(G.in_edges(vertex))/n_jobs)))
        
        with Parallel(n_jobs=n_jobs) as parallel:
            updated_ability = parallel(delayed(update)(neighbors, ability, f) for neighbors in neighbors_list)
        
            # Combine the updated abilities from all parallel workers
            for neighbor_ability in updated_ability:
                ability.update(neighbor_ability)

    return rank

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

    n_jobs = 5
    print("voterank")
    print(sorted(voterank(G).items(), key=lambda x: x[1]))
    print("parallel_voterank with {} jobs".format(n_jobs))
    print(sorted(parallel_voterank(G,n_jobs).items(), key=lambda x: x[1]))

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

    print("voterank")
    print(sorted(voterank(G).items(), key=lambda x: x[1]))
    print("parallel_voterank with {} jobs".format(n_jobs))
    print(sorted(parallel_voterank(G,n_jobs).items(), key=lambda x: x[1]))