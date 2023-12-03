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