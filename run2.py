import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import balanced_accuracy_score
from test import find_optimal_move
from calculate_return_rate import calculate_return_rate


temp = pd.read_csv("0050.TW-short.csv")
adj_close = temp["Adj Close"].values
df = pd.DataFrame({"Adj Close": adj_close})
features = []
df["Tomorrow"] = df["Adj Close"].shift(-1)

#calculate MA and RSI
window_size = [5, 10, 20, 50]
price_diff = df["Adj Close"].diff(1)
positive_price_changes = price_diff.where(price_diff > 0, 0)
negative_price_changes = -price_diff.where(price_diff < 0, 0)

for window in window_size:
    moving_average = df["Adj Close"].rolling(window).mean()
    name = "MA_ratio_{}".format(window)
    df[name] = df["Adj Close"] / moving_average
    features.append(name)

    average_gain = positive_price_changes.rolling(window).mean()
    average_loss = negative_price_changes.rolling(window).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    name = "RSI_window_{}".format(window)
    df[name] = rsi
    features.append(name)

df["Actions"] = find_optimal_move(df)
df = df.dropna()
target = "Actions"

total_rows = df.shape[0]
test_rows = 50
train_rows = total_rows - test_rows
train_df = df.iloc[:train_rows]
test_df = df.iloc[train_rows:]

def my_predict(model, features, train_df, test_df, buy_greater_than_sell_threshold = 0.1):
    model.fit(train_df[features], train_df[target])
    test_pred = model.predict(test_df[features])
    #return calculate_return_rate(adj_close, test_pred)
    test_pred = pd.Series(test_pred, index=test_df.index, name="Predictions")
    combined = pd.concat([test_df[target], test_pred], axis=1)
    return combined

def predict_with_prob(model, features, train_df, test_df, buy_greater_than_sell_threshold = 0.1):
    model.fit(train_df[features], train_df[target])

    #probability of [sell, hold, buy]
    test_pred = model.predict_proba(test_df[features])
    sell = test_pred[:, 0] #get first column
    hold = test_pred[:, 1] #get second column
    buy = test_pred[:, 2] #get third column
    #find the index of the column with the greatest probability
    max_value = np.argmax(test_pred, axis=1) 
    #if probability of buy is more than that of sell by a threshold, set action to buy
    max_value[(buy - sell) >= buy_greater_than_sell_threshold] = 2
    #0 = sell, 1 = hold, 2 = buy => -1 = sell, 0 = hold, 1 = buy
    max_values = max_value - 1
    test_pred = pd.Series(max_values, index=test_df.index, name="Predictions")

    combined = pd.concat([test_df[target], test_pred], axis=1)
    return combined



#exhasutive search to find the best threshold for the maximum score
'''

model = RandomForestClassifier(n_estimators=100, min_samples_split=2, min_samples_leaf = 2, random_state=666)
max_score = 0.0
best_threshold = -1
curr_threshold = 0.1
for threshold in np.arange(0.1, 0.5, 0.01):
    #predictions = backtest(df, model, features, threshold)
    predictions = predict(model, features, train_df, test_df, threshold)

    #predictions.to_csv("output.csv", index=None)
    #print(predictions["Predictions"].value_counts())
    #print(predictions["Actions"].value_counts())
    score = balanced_accuracy_score(predictions["Actions"], predictions["Predictions"])
    #score = precision_score(predictions["Actions"], predictions["Predictions"], average="micro")
    print(threshold, score)
    if score > max_score:
        best_threshold = threshold
        max_score = score
    #print(precision_score(predictions["Actions"], predictions["Predictions"], average="micro"))

print(best_threshold, max_score)
'''


param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 30, 50, 100, 200],
    'min_samples_split': [2, 5, 10, 50, 100, 200],
    'min_samples_leaf': [2, 5, 10, 20, 50, 100]
}


best_a = -1
best_b = -1
best_c = -1
best_d = -1
max_score = -1
for i in param_grid["n_estimators"]:
    for j in param_grid["max_depth"]:
        for k in param_grid["min_samples_split"]:
            for l in param_grid["min_samples_leaf"]:
                rf = RandomForestClassifier(n_estimators=i, max_depth=j, min_samples_split=k, min_samples_leaf=l, random_state=666)
                predictions = my_predict(rf, features, train_df, test_df)
                #score = my_predict(rf, features, train_df, test_df, 0.1)
                score = balanced_accuracy_score(predictions["Actions"], predictions["Predictions"])
                if score > max_score:
                    max_score = score
                    best_a = i
                    best_b = j
                    best_c = k
                    best_d = l
                    print(max_score, best_a, best_b, best_c, best_d)
                print(score, i, j, k, l)

print()
print(max_score, best_a, best_b, best_c, best_d)


#model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=666)
#test_pred, model = my_predict(model, features, train_df, test_df)
#print(test_pred)
#score = my_predict(model, features, train_df, test_df)
'''


mytrain_rows = total_rows - 40
mytrain_df = df.iloc[:mytrain_rows]
mytest_df = df.iloc[mytrain_rows:]
mytest_pred = model.predict(mytest_df[features])
#print(type(mytest_df[target]))
for i in range(len(mytest_pred)):
    print(mytest_pred[i])
print(df["Actions"].values[-40:])

joblib.dump(model, "best_model.pkl")
'''

joblib.dump(model, "best_model.pkl")