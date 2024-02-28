import numpy as np
import pandas as pd

def find_optimal_move(df: pd.DataFrame):
    prices = df["Adj Close"].values
    initial_capital = 1000
    stock_holdings = [[initial_capital / prices[0], 0]]
    cashes = [[initial_capital, 1]]
    for i in range(1, len(prices)):
        if stock_holdings[i-1][0] >= cashes[i-1][0] / prices[i]:
            stock_holdings.append([stock_holdings[i-1][0], 0])
        else:
            stock_holdings.append([cashes[i-1][0] / prices[i], 1])

        if cashes[i-1][0] >= stock_holdings[i-1][0] * prices[i]:
            cashes.append([cashes[i-1][0], 1])
        else:
            cashes.append([stock_holdings[i-1][0] * prices[i], 0])  

    #for i in range(len(prices)):
        #print(stock_holdings[i], cashes[i])

    pos = 1
    prev_pos = 1
    actions = np.zeros(len(prices))
    for i in range(len(prices) - 1, 0, -1):
        if pos == 1:
            prev_pos = cashes[i][1]
        elif pos == 0:
            prev_pos = stock_holdings[i][1]
        
        if prev_pos == 0 and pos == 1:
            actions[i] = -1
        elif prev_pos == 1 and pos == 0:
            actions[i] = 1
        else:
            actions[i] = 0
        pos = prev_pos

    if pos == 0:
        actions[0] = 1
    #print(actions)
    return actions