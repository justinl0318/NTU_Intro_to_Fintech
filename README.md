# NTU Intro_to_Fintech

### Stock price prediction of the 0050.TW ETF.

#### Model: Random forest classifier
Used dynamic programming to find the optimal action (buy, sell, hold) based on history price, and used them as labels to train the classifier. 

#### Features:
- MA (5, 10, 20, 25 days) 
- RSI (5, 10, 20, 25 days)

Used grid search to find the best parameters of the model.  
Output of model: the respective probability of [sell, hold, buy].  
Then, used exhastive search to find the optimal threshold for an action.

#### Evaluation metrics:
- balanced_accuracy_score
- precision_score

---

### Result
Test period: 2023/10/16 ~ 2023/12/22  
ROI: 6.7%
