import random

def find_most_important_agent_gamma_1(bids, N_prime, reports):
    # under the priority gamma_1, the priority of agent i is higher than agent j if:
    # 1. bids[i] > bids[j]
    # 2. bids[i] == bids[j] and len(reports[i]) > len(reports[j])
    # 3. bids[i] == bids[j] and len(reports[i]) == len(reports[j]) and i < j
    # return the most important agent
    prioritized_agents = sorted(N_prime, key=lambda x: (bids[x], len(reports[x]) if x in reports.keys() else 0, x), reverse=True)
    return prioritized_agents[0] if len(prioritized_agents) > 0 else None

def find_most_important_agent_gamma_2(bids, N_prime, reports):
    # under the priority gamma_2, the priority of agent i is higher than agent j if:
    return find_most_important_agent_gamma_1(bids, N_prime, reports)

def remove_exhaused_agents(bids, marked_not_exhausted, p_prime):
    to_remove = set()
    for agent in marked_not_exhausted:
        if bids[agent] < p_prime:
            to_remove.add(agent)
    marked_not_exhausted.difference_update(to_remove)

def find_first_critical_price(bids, marked_not_exhausted, k):
    L = sorted([bids[agent] for agent in marked_not_exhausted])
    change = {}
    for elem in L:
        if elem not in change.keys():
            change[elem] = 1
        else:
            change[elem] += 1
    # change -> {price: number of agents with that price}
    L = sorted(set(L))
    for value in L:
        sum_agents_before = sum(change[item] for item in L if L.index(item) < L.index(value))
        if sum_agents_before <= len(marked_not_exhausted) - k:
            return value 
    return None

def find_second_critical_price(bids, marked_not_exhausted, k):
    L = sorted([bids[agent] for agent in marked_not_exhausted])
    change = {}
    for elem in L:
        if elem not in change.keys():
            change[elem] = 1
        else:
            change[elem] += 1
    # change -> {price: number of agents with that price}
    L = sorted(set(L))
    for tentative_p in L:
        is_important = len(marked_not_exhausted) - 1 < k and k <= len(marked_not_exhausted)
        if not is_important:
            sum_agents_before = sum(change[item] for item in L if item > tentative_p)
            is_important = sum_agents_before - 1 < k and k <= sum_agents_before
            if is_important:
                return tentative_p
    return None
                
def snca(k, seller_net, reports, bids):
    
    allocation = {bidder: False for bidder in seller_net}
    payments = {bidder: 0 for bidder in seller_net}

    # seller = random.choice(list(seller_net))
    seller = 'a'
    unmarked = set(reports[seller]).union({seller})
    exhausted = set(seller)

    N_prime = set()
    if seller not in reports.keys():
        return allocation, payments

    p_prime = bids[seller]

    unmarked_exhausted = unmarked.intersection(exhausted)

    while len(unmarked_exhausted) > 0 and k > 0:

        print('unmarked_exhausted: ', unmarked_exhausted)

        for i in unmarked_exhausted:
            unmarked.remove(i)
            N_prime = N_prime.union(set(reports[i]))
            for neighbor in reports[i]:
                if bids[neighbor] < p_prime:
                    exhausted.add(neighbor)

        most_important_agent = find_most_important_agent_gamma_1(bids, N_prime.difference(exhausted), reports)
        print('most_important_agent: ', most_important_agent)

        if most_important_agent is not None:
            allocation[most_important_agent] = True
            payments[most_important_agent] = p_prime
            exhausted.add(most_important_agent)
            print(payments[most_important_agent])
            k -= 1

        unmarked_exhausted = unmarked.intersection(exhausted)

    marked_not_exhausted = unmarked.difference(exhausted)

    while len(marked_not_exhausted) > 0 and k > 0:

        demand = sum(1 for _ in marked_not_exhausted)
        # CASE 1: oversupplying
        if k >= demand:
            most_important_agent = find_most_important_agent_gamma_1(bids, marked_not_exhausted, reports)

        else:
            # CASE 2: undersupplying
            second_critical_price = find_second_critical_price(bids, marked_not_exhausted, k)
            print('second_critical_price: ', second_critical_price)
            if second_critical_price is not None:
                p_prime = second_critical_price
                remove_exhaused_agents(bids, marked_not_exhausted, p_prime)
                most_important_agent = find_most_important_agent_gamma_1(bids, marked_not_exhausted, reports)
                
            else:
                first_critical_price = find_first_critical_price(bids, marked_not_exhausted, k)
                print('first_critical_price: ', first_critical_price)
                if first_critical_price is not None:
                    p_prime = first_critical_price
                    remove_exhaused_agents(bids, marked_not_exhausted, p_prime)
                    most_important_agent = find_most_important_agent_gamma_2(bids, marked_not_exhausted, reports)
        
        print('most_important_agent: ', most_important_agent)
        allocation[most_important_agent] = True
        payments[most_important_agent] = p_prime
        marked_not_exhausted.remove(most_important_agent)
        k -= 1
    
    return allocation, payments

# test
if __name__ == '__main__':
    
    # test 1
    k = 4
    seller_net = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}

    # dense graph
    reports = {'a': {'b', 'c', 'd', 'e', 'f'}, 'b': {}, 'c': {}, 'd': {'e', 'f'}, 'e': {'f', 'g'}, 'f': {'g'}, 'g': {}}
    
    bids = {'a': 4, 'b': 5, 'c': 5, 'd': 6, 'e': 6, 'f': 7, 'g': 8}

    allocation, payments = snca(k, seller_net, reports, bids)
    print(allocation)
    print(payments)