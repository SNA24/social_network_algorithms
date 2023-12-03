import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
import networkx as nx

from utilities.priorityq import PriorityQueue

def reached(seller_net, reports):

    buyers = set()
    
    for bidder in seller_net:
        if bidder in reports.keys():
            buyers.update(reports[bidder])

    reached = set()

    for bidder in buyers:
        if bidder in reports.keys():
            reached.update(reports[bidder])

    buyers.update(reached)
    buyers.update(seller_net)

    return buyers

def vcg(k, seller_net, reports, bids):
    """
    # Parameters
    k : int
        Number of items to be sold.
    seller_net : set
        Set of strings, each identifying a different bidder.
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

    allocation = { bidder: False for bidder in bids.keys() }
    payments = { bidder: 0 for bidder in bids.keys() }

    # 1. ask bidders to announce their valuations for the items
    # OK conatained in bids

    # 2. choose a socially optimal assignment of items to bidders (a perfect matching that maximizes the total 
    # valuation of each buyer for what they get. This assignment is based on the announced valuations 
    # (since that’s all we have access to.))

    buyers = reached(seller_net, reports)
    perfect_matching = PriorityQueue()
    for buyer in buyers:
        perfect_matching.add(buyer, -bids[buyer])

    # 3. Charge each buyer the appropriate VCG price; that is, if buyer j receives item i under the optimal matching, then charge buyer j a 
    # price pij determined according to Equation p_{ij} = V_{B-j}^S - V_{B-j}^{S-i}.
    # Here, V_{B-j}^S is the total valuation of the set of buyers B-j for the set of items S, and V_{B-j}^{S-i} is the total valuation of the
    # set of buyers B-j for the set of items S-i, where S-i is the set of items S without item i.

    extracted = [ perfect_matching.pop() for _ in range(min(k+1, len(buyers))) ]
    V = -sum([ bids[bidder] for bidder in extracted[:-1] ])
    wannabe_winner = extracted[-1]

    for i in range(min(k, len(buyers))):
        winner = extracted[i]
        V_B_j_S = V + bids[winner] - bids[wannabe_winner]
        V_B_j_S_i = V + bids[winner]
        payments[winner] = V_B_j_S - V_B_j_S_i
        allocation[winner] = True

    return allocation, payments

# test
if __name__ == '__main__':
    
    # test 1
    k = 4
    seller_net = {'a', 'b'}

    reports = {'a': {'b', 'c'}, 'b': {'c','d'}, 'd': {'e', 'f', 'g'}, 'e': {'f', 'g'}, 'f': {'g'}}
    bids = {'a': 10, 'b': 109, 'c': 368, 'd': 12, 'e': 56, 'f': 25, 'g': 104}

    allocation, payments = vcg(k, seller_net, reports, bids)
    print(allocation)
    print(payments)

