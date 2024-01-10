import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utilities.priorityq2 import PriorityQueue
import random

MAX = 50

def update_agents(N, p, unmarked, exhausted, reports, bids):

    new_exhausted = set()
    for agent in unmarked:
        if bids[agent] < p:
            new_exhausted.add(agent)

    # Update the sets
    exhausted.update(new_exhausted)
    unmarked -= new_exhausted
    N -= new_exhausted

    # Add neighbors of newly exhausted agents to N_prime
    for agent in new_exhausted:
        if agent in reports:
            N.update(reports[agent])
            
def demand(N, p, bids, allocation, agent):
    # Demand is 1 if the agent's bid is at least p and they have not already been allocated an item
    return 1 if bids[agent] >= p and agent in N and allocation[agent] == False else 0
            
def check_important_agents(N, p, bids, allocation, k, reports):

    D_N_p = sum(demand(N, p, bids, allocation, agent) for agent in N)

    important_agents = PriorityQueue()
    res = False
    for agent in N:
        Di_N_p = demand(N, p, bids, allocation, agent)
        if D_N_p - Di_N_p < k <= D_N_p:
            res = True
            len_agent_reports = len(reports[agent]) if agent in reports else 0
            important_agents.put_with_priority((-bids[agent], -len_agent_reports), agent)
    
    return res, important_agents

def handle_important_agents(N, p, important_agents, allocation, payments, k, exhausted):
   
    if len(important_agents) == 0:
        k -= 1
        return allocation, payments, k, N, exhausted
    
    important_agent = important_agents.get_with_priority()[1]
    allocation[important_agent] = True
    payments[important_agent] = p  # Assuming the payment is the current price per item
    k -= 1  # Update the remaining number of items
        
    N.update(reports[important_agent] if important_agent in reports and important_agent not in exhausted else set())
    N.remove(important_agent)
    exhausted.add(important_agent)

    return allocation, payments, k, N, exhausted

def check_oversupplying_market(N, p, bids, allocation, k):
    return sum(demand(N, p, bids, allocation, agent) for agent in N) < k

def handle_oversupplying_market(N, p, bids, allocation, payments, k, exhausted):
    important_agents = PriorityQueue()
    for agent in N:
        important_agents.put_with_priority((-bids[agent], -len(reports[agent]) if agent in reports and agent not in exhausted else 0), agent)
    allocation, payments, k, N, exhausted = handle_important_agents(N, p, important_agents, allocation, payments, k, exhausted)
    return allocation, payments, k, N, exhausted

def find_i_critical_price(N, p, bids, k, allocation):
    p_star = p
    while p_star <= MAX:
        # Increase price by a small increment ε
        p_star += 1  # ε should be a small value
        D_N_p_star = sum(demand(N, p_star, bids, allocation, agent) for agent in N)
        if D_N_p_star <= k:
            break
    return p_star - 1  # The price just before demand drops below m

def find_ii_critical_price(N, p, bids, allocation, k, reports):
    p_star = p
    while p_star <= MAX:
        # Increase price by a small increment ε
        p_star += 1  # ε should be a small value
        if not check_important_agents(N, p_star, bids, allocation, k, reports)[0]:
            if check_important_agents(N, p_star + 1, bids, allocation, k, reports)[0]:
                return p_star
    return None # The price just before demand drops below m

def handle_undersupplying_market(N, p, bids, allocation, payments, k, reports, exhausted):
    
    p_star_ii = find_ii_critical_price(N, p, bids, allocation, k, reports)
    p_star_i = find_i_critical_price(N, p, bids, k, allocation)


    if p_star_ii is not None:
        # Use priority γ1 for II-critical price
        p = p_star_ii
        res, pq = check_important_agents(N, p+1, bids, allocation, k, reports)
        if res:
            allocation, payments, k, N, exhausted = handle_important_agents(N, p, pq, allocation, payments, k, exhausted)
        else:
            return None
    elif p_star_i is not None:
        # Use priority γ2 for I-critical price
        p = p_star_i
        important_agents = PriorityQueue()
        for agent in N:
            important_agents.put_with_priority((-bids[agent], -len(reports[agent]) if agent in reports and agent not in exhausted else 0), agent)
        allocation, payments, k, N, exhausted = handle_important_agents(N, p, important_agents, allocation, payments, k, exhausted)

    return N, p, allocation, payments, k, exhausted

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
    
    allocation = {bidder: False for bidder in set(bids.keys())}
    payments = {bidder: 0 for bidder in set(bids.keys())}

    N_prime = set(seller_net)
    p_prime = random.randint(0, MAX)

    unmarked = set(bids.keys())
    exhausted = set()
    
    while k > 0 and len(unmarked) > 0:
        
        update_agents(N_prime, p_prime, unmarked, exhausted, reports, bids)
        
        res, pq = check_important_agents(N_prime, p_prime, bids, allocation, k, reports)
        
        if res:

            allocation, payments, k, N_prime, exhausted = handle_important_agents(N_prime, p_prime, pq, allocation, payments, k, exhausted)
            
        elif check_oversupplying_market(N_prime, p_prime, bids, allocation, k):
            
            allocation, payments, k, N_prime, exhausted = handle_oversupplying_market(N_prime, p_prime, bids, allocation, payments, k, exhausted)

        elif not check_oversupplying_market(N_prime, p_prime, bids, allocation, k):

            res = handle_undersupplying_market(N_prime, p_prime, bids, allocation, payments, k, reports, exhausted) 
            
            if res is None:
                k -= 1
            else:
                N_prime, p_prime, allocation, payments, k, exhausted = res
                
        else:
            k -= 1
                 
    return allocation, payments
                
# test
if __name__ == '__main__':
    
    # test 1
    k = 6
    seller_net = {'a', 'b', 'c'}

    # dense graph
    reports = {'b': {'c'}, 'c': {'d', 'e'}, 'e': {'f'}, 'f': {'g'}}

    bids = {'a': 30, 'b': 14, 'c': 12, 'd': 16, 'e': 43, 'f': 27, 'g': 50}

    allocation, payments = snca(k, seller_net, reports, bids)
    print(allocation)
    print(payments)

        
