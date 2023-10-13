import networkx as nx
from priorityq import PriorityQueue
from lesson3 import betweenness

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

#The measure associated to each node is exactly its degree
def degree(G):
    cen=dict()
    for i in G.nodes():
        cen[i] = G.degree(i)
    return cen

#The measure associated to each node is the sum of the (shortest) distances of this node from each remaining node
#It is not exavtly the closeness measure, but it returns the same ranking on vertices
def closeness(G):
    cen=dict()

    for u in G.nodes():
        visited=set()
        visited.add(u)
        queue = [u]
        dist = dict()
        dist[u]  = 0
        while queue != []:
            v = queue.pop(0)
            for w in G[v]:
                if w not in visited:
                    queue.append(w)
                    visited.add(w)
                    dist[w] = dist[v] + 1
        cen[u] = sum(dist.values())
    return cen

#The measure associated to each node is its betweenness value
def btw(G):
    return betweenness(G)[1]

# PAGE RANK
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

# HITS
# G is assumed to be a directed graph
# steps and confidence are as for pageRank
# wa and wh are the weights that we assign to the authority value and the hub value of a node, in order to evaluate its rank
def hits(G, step=75, confidence=0, wa=0.5, wh=0.5):
    G = nx.DiGraph(G)
    time = 0
    n = nx.number_of_nodes(G)
    done = False

    hub = {i: float(1) / n for i in G.nodes()} # This contains the hub rank of each node.
    auth = {i: float(1) / n for i in G.nodes()} # This contains the authority rank of each node.

    while not done and time < step:
        time += 1

        htmp = {i:0 for i in G.nodes()} # it contains the new hub rank
        atmp = {i: 0 for i in G.nodes()} # it contains the new authority rank

        atot=0
        for i in G.nodes():
            for e in G.in_edges(i):
                # The authority level increases as better hubs are pointing to him
                atmp[i] += hub[e[0]] #the authority value of a node is the sum over all nodes pointing to him of their hubbiness value
                atot += hub[e[0]] #computes the sum of atmp[i] over all i. It serves only for normalization (each rank is done so that all values always sum to 1)

        htot=0
        for i in G.nodes():
            for e in G.out_edges(i):
                # The hubbiness level increases as it points to better authorities
                htmp[i] += auth[e[1]] #the hubbiness value of a node is the sum over all nodes at which it points of their authority value
                htot += auth[e[1]] #computes the sum of htmp[i] over all i. It serves only for normalization (each rank is done so that all values always sum to 1)

        adiff = 0
        hdiff = 0
        for i in G.nodes():
            adiff += abs(auth[i]-atmp[i] / atot)
            auth[i] = atmp[i] / atot
            hdiff += abs(hub[i] - htmp[i] / htot)
            hub[i] = htmp[i] / htot

        if adiff <= confidence and hdiff <=confidence:
            done = True

    return {i: wa*auth[i]+wh*hub[i] for i in G.nodes()}

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
    avg_deg = sum(G.degree(v) for v in G.nodes())/n # It computes the average degree of the network
    f = 1/avg_deg # It sets the decrement of weight in the network

    ability = {i : 1 for i in G.nodes()} # Initially the vote of each node weights 1
    for i in range(n):
        score = {i : 0 for i in G.nodes() if i not in rank.keys()}

        for u in score.keys():
            for v in G[u]:
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

# SHAPLEY DEGREE
# Compute the Shapley value for a characteristic function that extends degree centrality to coalitions.
# Specifically, the characteristic function is value(C) = |C| + |N(C)|, where N(C) is the set of nodes outside C with at least one neighbor in C.
# Even if the Shapley Value in general takes exponential time to be computed, for this particular characteristic function a polynomial time algorithm is known.
# Indeed, it has been proved that the Shapley value of node v in this case is SV[v] = 1/(1+deg(v)) + sum_{u \in N(v), u != v} 1/(1+deg(u)).
# For more information, see Michalack et al. (JAIR 2013) sec. 4.1
def shapley_degree(G):
    SV = {i:1/(1+G.degree(i)) for i in G.nodes()}
    for u in G.nodes():
        for v in G[u]:
            SV[u] += 1/(1+G.degree(v))

    return SV

# SHAPLEY THRESHOLD
# Consider another extension of degree centrality.
# Specifically, we assume that to influence a node outside a coalition is necessary that at least k of its neighbors are within the coalition.
# That is, the characteristic function is value(C) = |C| + |N(C,k)|, where N(C,k) is the set of nodes outside C with at least k neighbors in C.
# Even if the Shapley Value in general takes exponential time to be computed, for this particular characteristic function a polynomial time algorithm is known.
# Indeed, it has been proved that the Shapley value of node v in this case is SV[v] = min(1,k/(1+deg(v))) + sum_{u \in N(v), u != v} max(O,(deg(u)-k+1)/(deg(u)*(1+deg(u)))
# For more information, see Michalack et al. (JAIR 2013) sec. 4.2
def shapley_threshold(G, k=2):
    SV = {i:min(1,k/(1+G.degree(i))) for i in G.nodes()}
    for u in G.nodes():
        for v in G[u]:
            weight = max(0,(G.degree(u) - k + 1)/G.degree(u))
            SV[u] += weight * 1/(1+G.degree(v))
    return SV

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
    print("degree")
    print(top(G, degree, 3))
    print("closeness")
    print(top(G, closeness, 3))
    print("betweenness")
    print(top(G, btw, 3))
    print("pageRank")
    print(top(G, pageRank, 3))
    print("hits")
    print(top(G, hits, 3))
    print("voterank")
    print(top(G, voterank, 3))
    print("shapley_degree")
    print(top(G, shapley_degree, 3))
    print("shapley_threshold")
    print(top(G, shapley_threshold, 3))
