import random
from utilities.priorityq2 import PriorityQueue

def compute_demand(N_prime, p_prime, bids):
    return sum(1 for agent in N_prime if bids[agent] >= p_prime)

def compute_single_demand(p_prime, bid):
    return 1 if bid >= p_prime else 0

def compute_change_points(N_prime, bids):   
    change_points = PriorityQueue()
    for agent in N_prime:
        change_points.put_with_priority(bids[agent], agent)
    return change_points

def important_agents(N_prime, p_prime, bids, k, reports, exhausted):
    priority_gamma_1 = PriorityQueue()
    important_agents_found = set()
    total_demand = compute_demand(N_prime, p_prime, bids)
    for agent in N_prime:
        single_demand = compute_single_demand(p_prime, bids[agent])
        if total_demand - single_demand < k and k <= total_demand:
            important_agents_found.add(agent)
        if single_demand > 0 and agent not in exhausted:
            priority_gamma_1.put_with_priority((-bids[agent], -len(reports[agent]) if agent in reports.keys() else 0), agent)
    return important_agents_found, priority_gamma_1

def compute_priority_gamma_2(p_prime, p_prime_eps, N_prime, bids, reports, exhausted):
    priority_gamma_2 = PriorityQueue()
    for agent in N_prime:
        single_demand = compute_single_demand(p_prime, bids[agent])
        single_demand_eps = compute_single_demand(p_prime_eps, bids[agent])
        if agent not in exhausted and single_demand > 0:
            priority_gamma_2.put_with_priority((-single_demand_eps, -bids[agent], -len(reports[agent]) if agent in reports.keys() else 0), agent)
    return priority_gamma_2

def find_first_critical_price(change_points, N_prime, bids, k, reports, exhausted):

    prev_tentative_p = None
    prev_demand = None
    value = change_points.get_with_priority()
    while value:
        tentative_p = value[0]
        demand = compute_demand(N_prime, tentative_p, bids)
        if prev_demand is not None and k >= demand and k < prev_demand:
            return prev_tentative_p, compute_priority_gamma_2(prev_tentative_p, tentative_p, N_prime, bids, reports, exhausted)
        prev_tentative_p = tentative_p
        prev_demand = demand
        value = change_points.get_with_priority()

    return None, None

def find_second_critical_price(change_points, N_prime, bids, k, reports, exhausted):

    prev_tentative_p = None
    prev_gamma_1 = None
    value = change_points.get_with_priority()
    while value:
        tentative_p = value[0]
        important_agents_found, priority_gamma_1 = important_agents(N_prime, tentative_p, bids, k, reports, exhausted)
        if len(important_agents_found)>0 and prev_tentative_p is not None:
            return prev_tentative_p, prev_gamma_1
        elif len(important_agents_found)>0 and prev_tentative_p is None:
            break
        prev_tentative_p = tentative_p
        prev_gamma_1 = priority_gamma_1
        value = change_points.get_with_priority()
        
    return None, None

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

    seller = 's'
    N_prime = set(reports[seller]) if seller in reports.keys() else set()
    p_prime = bids[seller]

    unmarked = set(seller_net)
    unmarked.remove(seller)
    exhausted = set()
    
    while True:

        print('k: ', k)

        # check exhauseted
        for agent in N_prime:
            if bids[agent] < p_prime:
                exhausted.add(agent)

        # check unmarked
        for agent in N_prime:
            if agent in unmarked and agent in exhausted:
                N_prime = N_prime.union(set(reports[agent])) if agent in reports.keys() else N_prime
                unmarked.remove(agent)

        # terminate
        if k == 0 or (N_prime.intersection(unmarked) == set() and N_prime.intersection(exhausted) == N_prime):
            break

        # compute demand
        important_agents_found, priority_gamma_1 = important_agents(N_prime, p_prime, bids, k, reports, exhausted)
        demand = compute_demand(N_prime, p_prime, bids)

        # CONDITION 1 : IMPORTANT AGENTS - OVERSUPPLYING
        if len(important_agents_found) > 0 or k >= demand:
            print("CONDITION 1")
            agent = priority_gamma_1.get_with_priority()[2]

        # CONDITION 2 : UNDERSUPPLYING
        else:
            
            # CONDITION 2.1 : second critical price exists
            change_points = compute_change_points(N_prime, bids)

            second_critical_price, priority_gamma_1 = find_second_critical_price(change_points.copy(), N_prime, bids, k, reports, exhausted)

            if second_critical_price is not None and len(priority_gamma_1) > 0:
                print("CONDITION 2.1")
                p_prime = second_critical_price
                agent = priority_gamma_1.get_with_priority()[2]

            # CONDITION 2.2 : first critical price exists
            else:

                print("CONDITION 2.2")

                first_critical_price, priority_gamma_2 = find_first_critical_price(change_points.copy(), N_prime, bids, k, reports, exhausted)

                p_prime = first_critical_price
                if len(priority_gamma_2) > 0:
                    agent = priority_gamma_2.get_with_priority()[2]
                else:
                    break

        allocation[agent] = True
        payments[agent] = p_prime
        k -= 1
        exhausted.add(agent)
        print("winner: ", agent, " price: ", p_prime)

    return allocation, payments
                
# test
if __name__ == '__main__':
    
    # test 1
    k = 4
    seller_net = {'s', 'a', 'b', 'c', 'd', 'e', 'f', 'g'}

    # dense graph
    reports = {'s': {'a','b'}, 'b': {'c'}, 'c': {'d', 'e'}, 'e': {'f'}, 'f': {'g'}}

    bids = {'s': 2, 'a': 3, 'b': 1, 'c': 1, 'd': 6, 'e': 4, 'f': 7, 'g': 5}

    allocation, payments = snca(k, seller_net, reports, bids)
    print(allocation)
    print(payments)

        
