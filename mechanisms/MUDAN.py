import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utilities.priorityq import PriorityQueue

def mudan(k, seller_net, reports, bids):
    """
    # Parameters
    k : int
        Number of items to be sold.
    seller_net : set
        Set of strings, each idnetifying a different bidder.
    reports : dict
        Dictionary whose keys are strings each identifying a different bidder and whose
        values are sets of strings representing the set of bidders to which the bidder identified by the
        key reports the information about the auction.
    bids : dict
        Dictionary whose keys are strings each identifying a different bidder and whose
        values are numbers defining the bid of the bidder identified by that key.

    # Returns
    allocation : dict
        Dictionary that has as keys the strings identifying each of the bidders
        that submitted a bid, and as value a boolean True if this bidder is allocated one of the items,
        and False otherwise.
    payments : dict
        Dictionary that has as keys the strings identifying each of the bidders that
        submitted a bid, and as value the price that she pays. Here, a positive price 
        means that the bidder is paying to the seller, while a negative price means that 
        the seller is paying to the bidder.
    """
    # Implement the MUDAN mechanism based on Fang et al. (2023) or your specific algorithm

    allocation = { bidder : False for bidder in bids.keys() }
    payments = { bidder : 0 for bidder in bids.keys() }

    W = set() # winner set
    A = seller_net

    if len(A) == 0:
        return allocation, payments
    
    valuations = PriorityQueue()
    for bidder in A.difference(W):
        valuations.add(bidder, -bids[bidder])
    
    marked = set() # marked set
    # init P, potential winner set
    if len(A.difference(W)) <= k:
        P = A
    else:
        P = W.union(set([valuations.pop() for _ in range(k)]))

    # while the difference between P and W is different from the empty set, do
    while len(P.difference(W)) > 0:
        # while A contains an unmarked agent i \in W \union (A \difference P), do
        unmarked_agents = A.intersection(W.union(A.difference(P))).difference(marked)

        for unmarked_agent in unmarked_agents:
            # update A with A \union r_i', mark the agent i 
            if unmarked_agent in reports.keys():
                A = A.union(reports[unmarked_agent])
            marked.add(unmarked_agent)

        # update P
        if len(A.difference(W)) <= k:
            P = A
        else:
            valuations = PriorityQueue()
            for bidder in A.difference(W):
                valuations.add(bidder, -bids[bidder])
            P = W.union(set([valuations.pop() for _ in range(k)]))

        # assign a priority \sigma to each agent in P
        sigma = PriorityQueue()
        diff_PW = P.difference(W)
        for agent in P:
            if agent in diff_PW:
                sigma.add(agent, -len(reports[agent]) if agent in reports.keys() else 0)
        diff_AW = A.difference(W)
        # add the w element in diff_PW with the highest priority in W
        w = sigma.pop()
        W.add(w)
        # record the tentative payment of w
        valuations = PriorityQueue()
        for bidder in A.difference(W):
            valuations.add(bidder, -bids[bidder])
        if len(diff_AW) > k:
            for _ in range(k-1):
                valuations.pop()
            try:
                payments[w] = bids[valuations.pop()]
            except:
                payments[w] = 0
        else:
            payments[w] = 0
        allocation[w] = True
        k -= 1

    return allocation, payments

# test
if __name__ == '__main__':
    
    k = 4
    seller_net = {'a', 'b'}

    reports = {'b': {'c'}, 'c': {'d', 'e'}, 'e': {'f'}, 'f': {'g'}}

    bids = {'a': 3, 'b': 1, 'c': 1, 'd': 6, 'e': 4, 'f': 7, 'g': 5}

    allocation, payments = mudan(k, seller_net, reports, bids)
    print(allocation)
    print(payments)