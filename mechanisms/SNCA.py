import random 

def find_important_bidder(bids, N_prime, p_prime, reports):

    important_bidders = {}
    for bidder in N_prime:
        # if is important add the bidder and its bid to the dictionary
        if (bids[bidder] >= p_prime):
            important_bidders[bidder] = bids[bidder]

    # sort the dictionary by value and return the first element
    # TODO: check if the values are equals and sort by number of neighbors in such a case
    important_bidders = sorted(important_bidders.items(), key=lambda x: x[1], reverse=True)

    # take the bidders with the highest bid
    important_bidders = [bidder for bidder in important_bidders if bidder[1] == important_bidders[0][1]]

    # if there is one or more bidders with the same bid, sort them by number of neighbors
    return sorted(important_bidders, key=lambda x: len(reports[x[0]]) if x[0] in reports.keys() else 0, reverse=True)

def snca(k, seller_net, reports, bids):
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

    allocation = {bidder: False for bidder in seller_net}
    payments = {bidder: 0 for bidder in seller_net}

    # choose random seller
    seller = random.choice(list(seller_net))
    # seller = 'b'

    if seller not in reports.keys():
        return allocation, payments
    
    p_prime = bids[seller]
    N_prime = set()
    for neighbor in reports[seller]:
        if bids[neighbor] >= p_prime:
            N_prime.add(neighbor)
    
    i = 0

    while (len(N_prime) > 0 and k > 0):

        tot_demand = sum(1 for bidder in N_prime if bids[bidder] >= p_prime)

        if (k < tot_demand):
            # UNDERSUPPLYING
            available_bids = sorted([bids[bidder] for bidder in N_prime if bids[bidder] > p_prime], reverse=True)
            # take all the bids with the highest value
            available_bids = [bid for bid in available_bids if bid == available_bids[0]]
            # sort them by number of neighbors
            available_bids = sorted(available_bids, key=lambda x: len(reports[x]) if x in reports.keys() else 0, reverse=True)
            
            if len(available_bids) > 1:
                p = available_bids[1]
            else:
                p = available_bids[0]   

        else: 
            # OVERSUPPLYING
            p = p_prime
        
        # detect important agents
        important_bidders = find_important_bidder(bids, N_prime, p, reports)

        if (len(important_bidders) > 0):

            important_bidder = important_bidders[0][0]
            payments[important_bidder] = p
            allocation[important_bidder] = True
            k -= 1

            # update the circumstance
            if important_bidder in reports.keys():
                for neighbor in reports[important_bidder]:
                    if neighbor not in payments.keys():
                        if bids[neighbor] >= p:
                            N_prime.add(neighbor)
            N_prime.remove(important_bidder)

    return allocation, payments

# test
if __name__ == '__main__':
    
    # test 1
    k = 4
    seller_net = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}

    # dense graph
    reports = {'a': {'b', 'c', 'd', 'e', 'f', 'g'}, 'b': {'a', 'c', 'd', 'e', 'f', 'g'}, 'c': {'a', 'b', 'd', 'e', 'f', 'g'}, 'd': {'a', 'b', 'c', 'e', 'f', 'g'}, 'e': {'a', 'b', 'c', 'd', 'f', 'g'}, 'f': {'a', 'b', 'c', 'd', 'e', 'g'}, 'g': {'a', 'b', 'c', 'd', 'e', 'f'}}
    
    bids = {'a': 10, 'b': 109, 'c': 368, 'd': 12, 'e': 56, 'f': 25, 'g': 104}

    allocation, payments = snca(k, seller_net, reports, bids)
    print(allocation)
    print(payments)