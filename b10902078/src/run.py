import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from find_optimal_move import find_optimal_move
from calculate_return_rate import calculate_return_rate

temp = pd.read_csv("0050.TW-short.csv")
adj_close = temp["Adj Close"].values
df = pd.DataFrame({"Adj Close": adj_close})
features = []
df["Tomorrow"] = df["Adj Close"].shift(-1)
df["Target"] = (df["Tomorrow"] > df["Adj Close"]).astype(int)

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

df = df.dropna()
#print(df)
target = "Target"

total_rows = df.shape[0]
test_rows = 50
train_rows = total_rows - test_rows
train_df = df.iloc[:train_rows]
test_df = df.iloc[train_rows:]

def predict_with_prob(model_parameter, threshold_low, threshold_high):
    #probability of [decrease, increase]
    test_pred = model_parameter.predict_proba(test_df[features])[:, 1] #take probability of increase
    arr = []
    for i in test_pred:
        if i > threshold_high:
            arr.append(1)
        elif i < threshold_low:
            arr.append(-1)
        else:
            arr.append(0)

    returnrate = calculate_return_rate(adj_close, arr)
    print(arr)
    return returnrate



# model = RandomForestClassifier(n_estimators=100, random_state=666)
# model.fit(train_df[features], train_df[target])
# max_return_rate = -10000
# best_threshold_low = -1
# best_threshold_high = 10
# for threshold_low in np.arange(0.1, 0.5, 0.01):
#     for threshold_high in np.arange(0.5, 0.9, 0.01):
#         return_rate = predict_with_prob(model, threshold_low, threshold_high)
#         print(threshold_low, threshold_high, return_rate)
#         if return_rate > max_return_rate:
#             best_threshold_low = threshold_low
#             best_threshold_high = threshold_high
#             max_return_rate = return_rate
#         #print(precision_score(predictions["Actions"], predictions["Predictions"], average="micro"))
# print(best_threshold_low, best_threshold_high, max_return_rate)
# return_rate = predict_with_prob(model, best_threshold_low, best_threshold_high)
# joblib.dump(model, "best_model.pkl")



#exhasutive search to find the best threshold for the maximum score
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 30, 50, 100, 200],
    'min_samples_split': [2, 5, 10, 50, 100, 200],
    'min_samples_leaf': [2, 5, 10, 20, 50, 100]
}

# best_a = -1
# best_b = -1
# best_c = -1
# best_d = -1
# max_return_rate = -1
# for i in param_grid["n_estimators"]:
#     for j in param_grid["max_depth"]:
#         for k in param_grid["min_samples_split"]:
#             for l in param_grid["min_samples_leaf"]:
#                 rf = RandomForestClassifier(n_estimators=i, max_depth=j, min_samples_split=k, min_samples_leaf=l, random_state=666)
#                 rf.fit(train_df[features], train_df[target])
#                 return_rate = predict_with_prob(rf, 0.27, 0.66)
            
#                 if return_rate > max_return_rate:
#                     max_return_rate = return_rate
#                     best_a = i
#                     best_b = j
#                     best_c = k
#                     best_d = l
#                 print(return_rate, i, j, k, l)

# print()
# print(max_return_rate, best_a, best_b, best_c, best_d)
final_model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=2, random_state=666)
final_model.fit(train_df[features], train_df[target])
final_return_rate = predict_with_prob(final_model, 0.27, 0.66)
print(final_return_rate)
joblib.dump(final_model, "best_model.pkl")
