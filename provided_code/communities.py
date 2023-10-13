import networkx as nx
import itertools as it

#######STRONGLY CONNECTED COMPONENTS [Leskovec et al., Sect. 10.8.6]#######

# The standard algorithm for computing Strongly Connected Components
# It runs a DFS on the original graph and saves the order in which nodes have been exhausted (all neighbors have been visited).
# Next runs another DFS on the reversed graph by considering nodes in the order opposite to the one in which they are exhausted.
# Since it just requires to run two DFS, then its complexity is simply O(n+m)
def strongly(G):
    # First DFS on the original graph
    visited = [] #It keeps the set of all visited nodes
    order = [] #It keeps the order in which nodes are exhausted
    while len(visited) < len(G.nodes()): # Until all nodes have been visited
        root = next(node for node in G.nodes() if node not in visited) # Returns the first node that has not been yet visited
        stack = [root] # It is the stack used by DFS (we here provide an iterative implementation)
        visited.append(root)
        while len(stack) > 0:
            # We will consider the node on the top of the stack.
            # Note that we are not popping it out.
            # We only pop out this node when it is exhausted
            u = stack[-1]
            try: # We check if the node is exhausted, i.e. it has not unvisited neighbors
                v = next(v for v in G[u] if v not in visited)  # If u is exhausted, then a StopIteration exception is raised
            except: # If u is exhausted
                order.append(u)
                stack.pop()
            else: # If u is not exhausted, we visit its next neighbor
                stack.append(v)
                visited.append(v)

    # We now run a DFS on the reversed graph by considering vertices according to the reverse of order
    invG = G.reverse(False)  # The graph with reversed edges
    components = [] # It will keep the found components
    i = 0 # It will keep the number of found components
    ivisited = [] # It will keep the set of visited nodes in this second visit
    while len(ivisited) < len(visited): # Until all nodes have been visited
            stack = [next(node for node in order[::-1] if node not in ivisited )] # Start from the first unvisited node according to the reverse of order
            components.append([])
            # Run a DFS
            while len(stack) > 0:
                # We can now simply pop out the node, since we do not need to mark when it is exhausted
                u = stack.pop()
                ivisited.append(u)
                components[i].append(u)
                for v in invG[u]:
                    if v not in ivisited:
                        stack.append(v)
            i += 1

    return components

# Above algorithm turns out to be hard to parallelize, since the order in which nodes are exhausted is vital
# for the second DFS to return the right components.
# Next we present a potentially slower algorithm that turns out to be easier to parallelize.
# The idea is to run a DFS visit from the same node u on both the original and reversed graph:
# the nodes visited in both visits are the ones belonging to the same component of u.
# We can now replace all these nodes in the graph with a unique super-node, and repeating from another node in the graph.
# We will end up with a graph containing only super-nodes, each representing a different component.
# Hence, this algorithm makes two DFS visits for a number of times that is equivalent to the number of strongly connected components.
# However, this approach can be easier to parallelize, since different jobs can be assigned different portion of a graph,
# so that they compute strongly connected components in each portion, and return the corresponding super-nodes (and their edges)
# We can next re-run the algorithm on the graph consisting of returned super-nodes in order to verify if these can be merged
# in larger strongly connected components.
def strongly2(G):
    graph = G.copy() # We will work on a copy of the original graph, since we are going to modify the graph
    components = [] # It saves the components that will be returned
    comps = dict() # It saves the components by indexing each component by its root

    done = False
    # We will iteratively try to merge nodes, until we are able to do it.
    # Once no nodes exists that can be merged in a super-node, we go out from this while loop
    while not done:
        changed = False

        # In each iteration, we try merging from every possible node
        for node in graph.nodes():
            # First DFS on the original graph
            visited = set()
            queue = [node]
            while len(queue) > 0:
                u = queue.pop()
                for v in graph[u]:
                    if v not in visited:
                        queue.append(v)
                visited.add(u)

            igraph=graph.reverse(False) # The reversed graph
            # Second DFS on the reversed graph
            ivisited = set()
            queue = [node]
            while len(queue) > 0:
                u = queue.pop()
                for v in igraph[u]:
                    if v not in ivisited:
                        queue.append(v)
                ivisited.add(u)

            # visited & ivisited is the strongly connected component of node
            if len(visited&ivisited) > 1: # If this component has size larger than 1, then merging is needed
                comps[node]=visited&ivisited

                mapping = {k:node for k in comps[node]} # We will rename each member of the component with node
                graph = nx.relabel_nodes(graph,mapping, copy=False) # This networkx function execute the renaming
                if graph.has_edge(node, node): # The relabeling may create a loop for the supernode
                    graph.remove_edge(node, node) # In this case we remove the loop

                changed = True # If merging occurs, then this iteration has not been merge-less and a new iteration is necessary
                break

        if not changed: # If no merging occurred (for any possible starting point), then we can leave the while loop
            done = True

    # Every super-node corresponds to a component.
    # Moreover, every never-merged node corresponds to a singleton component.
    for u in graph.nodes():
        if u in comps.keys():
            components.append(comps[u])
        else:
            components.append({u})

    return components


#######DISCOVER COMPLETE BIPARTITE SUBGRAPH [Leskovec et al., Sect. 10.3, 6.2.6]#######

# The following is an algorithm born to discover which items are usually sold together.
# Specifically, it takes on input a set of baskets B, each containing some items from the universe set N.
# We will then ask for a set of t items that are contained in at least s different baskets.
# The naive algorithm looks for all subsets of N of size t, and counts how many baskets contain each subset.
# Unfortunately, this algorithm turns out to be very inefficient as long as t increases.
# The algorithm below instead use the following idea to filter out some subsets for each size i < t:
# Given a subset S of size i, this can be contained in s baskets only if
# all subsets of S of size i-1 are contained in s baskets.
# Hence, by saving the subsets of size i-1 that are contained in s baskets,
# one may focus only on the sets of size i that contain them as subsets,
# and count only the number of baskets containing these sets of items (and not each possible set of items).
# Usually, for i small many subsets pass the filter, but the number of subsets to check is not very large,
# while fot large i few subsets pass the filter among the many possible subsets.
def apriori(B, N, s, t):
    sol = dict() # It will keep as keys the set of t items that are contained in at least s baskets, and as value the at least s baskets containing them

    # We will build our sets incrementally starting from i = 1
    # For each i, we will consider all possible candidate sets of i items (that is, the ones for which all subsets of size i-1 have not been excluded)
    # Then for each candidate set, we count the number of baskets containing it, and we exclude the ones contained in less than s baskets
    for i in range(t):
        cand = dict() # It will keep for each candidate subset of size i the baskets containing it

        if i == 0:  # At beginning, we consider all possible singletons as candidate set
            live_cand = it.combinations(N,1)
        else:   # In next iterations, we only consider those subsets whose subsets of size i-1 have not been excluded
            # How to do it?
            # Consider a not excluded subset T of size i-1
            # For each element x that is not in T, try to add x to T, and check if all subsets S of T+x are not excluded
            live_cand = []
            for T in not_excluded:
                for x in N:
                    if x not in T:
                        candidate = T+(x,)
                        passed = True
                        for S in it.combinations(candidate,i):
                            if S not in not_excluded:
                                passed = False
                                break
                        if passed and candidate not in live_cand:
                            live_cand.append(candidate)

        not_excluded=[]
        # For all candidates, we evaluate the set of baskets containing them
        for S in live_cand:
            cand[S] = []
            for j in B.keys():
                if set(S) <= set(B[j]): # if S is a subset of B[j]
                    cand[S].append(j)

            # We will keep as not excluded only those candidates that are contained in at least s baskets
            if len(cand[S]) >= s:
                not_excluded.append(S)

    # After t iterations, in not excluded there are subsets of t items that are contained in at least s baskets
    for S in not_excluded:
        sol[S]=cand[S] # cand[S] contains the (at least s) baskets that contain S

    return sol

# We solve the problem of finding complete (s,t)-bipartite subgraphs as follows:
# Let items be the nodes of a graph, and baskets be the neighborhood of nodes,
# then complete (s, t)-bipartite subgraphs corresponds to sets of t items that are contained in at least s baskets
def Kst(G, s, t):
    # Building baskets
    B = dict()
    for i in G.nodes():
        B[i] = set(G[i])

    # Computing the solution
    sol = apriori(B, set(B.keys()), s, t)

    # Solution consists of a dictionary of (items, baskets)
    # We next translate these pairs in set of nodes: this will help also to remove redundancies
    # (e.g., a clique of 4 nodes will be returned 4 times by the a priori algorithm, but they corresponds at the same sets of nodes)
    community = set()
    for S in sol.keys():
        # Only hashable (and thus immutable) objects can be included in a set, thus we use a frozenset in place of a set
        community.add(frozenset(sol[S]+list(S)))

    return community


# How to choose s and t such that we are almost sure to find at least one community with the desired properties?
# Suppose first that the network is d-regular, that is all nodes have degree exactly d
# Given a subset T of t nodes, we would like to compute the expected number of nodes that are neighbors of all nodes in T
# The probability that a node u in T is in the neighborhood of a node v is d/n
# The probability that all nodes of T are in the neighborhood of a node v is (d/n)^t
# Then expected number of nodes that have T in the neighborhood is n(d/n)^t
# Hence, if this expected number is at least s, then it is highly probable that an (s,t)-complete bipartite subgraph exists
#
# If the network is not d-regular, then the discussion above can be repeated with d being the average degree.
#
# In conclusion, we choose s and t such that n (d/n)^t > s.

###############################################

if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(3, 6)
    G.add_edge(4, 5)
    G.add_edge(4, 7)
    G.add_edge(5, 2)
    G.add_edge(5, 7)
    G.add_edge(6, 3)
    G.add_edge(6, 8)
    G.add_edge(7, 8)
    G.add_edge(7, 10)
    G.add_edge(8, 7)
    G.add_edge(9, 7)
    G.add_edge(10, 9)
    G.add_edge(10, 11)
    G.add_edge(11, 12)
    G.add_edge(12, 10)
    print(strongly(G))
    print(strongly2(G))

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
    print(list(nx.find_cliques(G)))
    print(Kst(G, 1, 3))
    print(Kst(G, 2, 2))
    print(Kst(G, 2, 3))
