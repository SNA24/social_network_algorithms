import random

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

    tentative_payments = {}

    #seller = random.choice(list(seller_net))
    seller = 's'
    print("seller: ", seller)

    W = set() # winner set
    A = reports[seller] if seller in reports.keys() else set() 
    marked = set() # marked set
    if len(A) == 0:
        return allocation, payments
    # init P, potential winner set
    if len(A.difference(W)) <= k:
        P = A
    else:
        # P := W ∪ {i_1,...,i_m′}
        P = W.union(set([item[0] for item in sorted(bids.items(), key=lambda x: x[1], reverse=True) if item[0] in A.difference(W)][0:k]))

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
                # P := W ∪ {i_1,...,i_m′}
                P = W.union(set([item[0] for item in sorted(bids.items(), key=lambda x: x[1], reverse=True) if item[0] in A.difference(W)][0:k]))

        # assign a priority \sigma to each agent in P
        sigma = { agent: len(reports[agent]) if agent in reports.keys() else 0 for agent in P }
        diff_PW = P.difference(W)
        diff_AW = A.difference(W)
        # take the w elementi in siff_PW with the highest priority in W
        w = sorted(diff_PW, key=lambda x: sigma[x], reverse=True)[0]
        W.add(w[0])
        # record the tentative payment of w
        sorted_valuation = sorted([bids[agent] for agent in diff_AW], reverse=True)
        if len(sorted_valuation) > k:
            tentative_payments[w] = sorted_valuation[k]
        else:
            tentative_payments[w] = 0
        k -= 1

    # determine the allocation and payment results using W and tentative payments
    for bidder in tentative_payments.keys():
        allocation[bidder] = True
        payments[bidder] = tentative_payments[bidder]

    return allocation, payments

# test
if __name__ == '__main__':
    
    # test 1
    k = 4
    seller_net = {'s', 'a', 'b', 'c', 'd', 'e', 'f', 'g'}

    # dense graph
    reports = {'s': {'a','b'}, 'b': {'c'}, 'c': {'d', 'e'}, 'e': {'f'}, 'f': {'g'}}

    bids = {'a': 3, 'b': 1, 'c': 1, 'd': 6, 'e': 4, 'f': 7, 'g': 5}

    allocation, payments = mudan(k, seller_net, reports, bids)
    print(allocation)
    print(payments)