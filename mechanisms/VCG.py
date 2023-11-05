def vcg(k, seller_net, reports, bids):
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

    allocation = {}
    payments = {}
    
    return allocation, payments
