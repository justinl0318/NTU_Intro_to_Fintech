import numpy as np

def calculate_return_rate(priceVec, suggested_action):
    capital=1000	# Initial available capital
    capitalOrig=capital		# original capital
    dataCount=len(priceVec)				# day size
    initial_datacount = dataCount - 50
    suggestedAction = np.zeros((dataCount, 1))
    #suggestedAction=np.zeros((dataCount,1))	# Vec of suggested actions
    stockHolding=np.zeros((dataCount,1))  	# Vec of stock holdings
    total=np.zeros((dataCount,1))	 	# Vec of total asset
    realAction=np.zeros((dataCount,1))	# Real action 
    index = 0
    for ic in range(initial_datacount, dataCount):
        currentPrice=priceVec[ic]	# current price
        suggestedAction[ic] = [suggested_action[index]]
        #print(suggested_action[index], suggestedAction[ic])
        index += 1
        if ic>0:
            stockHolding[ic]=stockHolding[ic-1]	# The stock holding from the previous day
        if suggestedAction[ic]==1:	# Suggested action is "buy"
            if stockHolding[ic]==0:		# "buy" only if you don't have stock holding
                stockHolding[ic]=capital/currentPrice # Buy stock using cash
                capital=0	# Cash
                realAction[ic]=1
        elif suggestedAction[ic]==-1:	# Suggested action is "sell"
            if stockHolding[ic]>0:		# "sell" only if you have stock holding
                capital=stockHolding[ic]*currentPrice # Sell stock to have cash
                stockHolding[ic]=0	# Stocking holding
                realAction[ic]=-1
        elif suggestedAction[ic]==0:	# No action
            realAction[ic]=0
        else:
            assert False
        total[ic]=capital+stockHolding[ic]*currentPrice	# Total asset, including stock holding and cash 
    returnRate=(total[-1]-capitalOrig)/capitalOrig		# Return rate of this run
    return returnRate