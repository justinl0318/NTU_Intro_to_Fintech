def myStrategy(pastPriceVec, currentPrice):
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    df = pd.DataFrame({"Adj Close": pastPriceVec})
    df.loc[len(df)] = {"Adj Close": currentPrice}
    
    features = []

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

    #df["Actions"] = find_optimal_move(df)
    #print(df)
    df = df.dropna()
    if (len(df) == 0):
        #print(len(df), 0)
        return 0
    
    threshold_low = 0.27
    threshold_high = 0.66
    model = joblib.load("src/best_model.pkl")
    target = "Actions"
    test_df = df.tail(1)
    test_pred = model.predict_proba(test_df[features])[:, 1] #take probability of increase
    arr = []
    for i in test_pred:
        if i > threshold_high:
            arr.append(1)
        elif i < threshold_low:
            arr.append(-1)
        else:
            arr.append(0)
    return arr[0]



