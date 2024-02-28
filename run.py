import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("0050.TW-short.csv")
total_rows = df.shape[0]
#print(df)
df.plot.line(y="Close", use_index=True)
df["Action"] = 0

#dp to find maximum profit based on historical price
#pos: 0 = at stock, 1 = at cash
initial_capital = 1000
stock_holdings = [[initial_capital / df["Adj Close"][0], 0]]
cashes = [[initial_capital, 1]]
for i in range(1, total_rows):
    if stock_holdings[i-1][0] >= cashes[i-1][0] / df["Adj Close"][i]:
        stock_holdings.append([stock_holdings[i-1][0], 0])
    else:
        stock_holdings.append([cashes[i-1][0] / df["Adj Close"][i], 1])

    if cashes[i-1][0] >= stock_holdings[i-1][0] * df["Adj Close"][i]:
        cashes.append([cashes[i-1][0], 1])
    else:
        cashes.append([stock_holdings[i-1][0] * df["Adj Close"][i], 0])   

#backtracking
pos = 1 #start from the back = cash => pos = 1
prev_pos = 1
for i in range(total_rows - 1, 0, -1):
    #check previous location
    if pos == 1: 
        prev_pos = cashes[i][1]
    elif pos == 0:
        prev_pos = stock_holdings[i][1]
    
    #if prev_pos is at stock and curr_pos is at cash, that means we sold at this day
    if prev_pos == 0 and pos == 1:
        df.loc[i, "Action"] = -1 #sell
    #if prev_pos is at cash and curr_pos is at stock, that means we buy at this day
    elif prev_pos == 1 and pos == 0:
        df.loc[i, "Action"] = 1 #buy
    else:
        df.loc[i, "Action"] = 0
    #update pos
    pos = prev_pos

#if we start from stock, that means we buy on the first day
if pos == 0:
    df.loc[0, "Action"] = 1

#print(df)


model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=666)

features = ["Adj Close", "Volume", "Open", "High", "Low"]
target = ["Action"]
train_rows = int(0.8 * total_rows)
test_rows = total_rows - train_rows
train_df = df.iloc[:train_rows]
test_df = df.iloc[train_rows:]
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]
print(y_train)
#X = feature, y = target
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
print(accuracy)



