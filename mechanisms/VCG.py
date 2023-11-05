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

    allocation = {}
    payments = {}

    # 1. ask bidders to announce their valuations for the items
    # OK conatained in bids

    # 2. choose a socially optimal assignment of items to bidders (a perfect matching that maximizes the total 
    # valuation of each buyer for what they get. This assignment is based on the announced valuations 
    # (since that’s all we have access to.))

    sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)

    winners_list = [bid[0] for bid in sorted_bids]
    # remove from winners list the nodes which are in none of the report values 
    for winner in winners_list:
        find = False
        for report in reports.values():
            if winner in report:
                find = True
                break
        if not find:
            winners_list.remove(winner)

    print(winners_list)

    # 3. Charge each buyer the appropriate VCG price; that is, if buyer j receives item i under the optimal matching,
    # then charge buyer j a price pij determined according to Equation pij = sw(A-j) - sw(A-j-i) where sw(A-j) is the
    # social welfare of the optimal matching A, A-j-i is the optimal matching that results if we remove item i from
    # A and buyer j from A, sw(A-j) is the social welfare of the optimal matching that results if we remove buyer j
    # from A.

    for bidder in seller_net:

        if bidder in winners_list[:min(k, len(winners_list))]:

            allocation[bidder] = True

            # compute the social welfare without the bidder but with the slot
            sw_without_bidder = 0

            for winner in winners_list[:min(k+1, len(winners_list))]:
                if winner != bidder:
                    sw_without_bidder += bids[winner]

            # compute the social welfare without the bidder and the slot

            sw_without_bidder_and_slot = 0

            for winner in winners_list[:min(k, len(winners_list))]:
                if winner != bidder:
                    print(winner)
                    sw_without_bidder_and_slot += bids[winner]

            print(sw_without_bidder, sw_without_bidder_and_slot)

            payments[bidder] = sw_without_bidder - sw_without_bidder_and_slot

        else :
            allocation[bidder] = False
            payments[bidder] = 0

    return allocation, payments

# test
if __name__ == '__main__':
    
    # test 1
    k = 4
    seller_net = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}

    reports = {'a': {'b', 'c'}, 'b': {'c', 'd'}, 'd': {'e', 'f', 'g'}, 'e': {'f', 'g'}, 'f': {'g'}}
    bids = {'a': 10, 'b': 109, 'c': 368, 'd': 12, 'e': 56, 'f': 25, 'g': 104}

    allocation, payments = vcg(k, seller_net, reports, bids)
    print(allocation)
    print(payments)

