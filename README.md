# Customer Churn Prediction

This project uses Python to predict whether a telecom customer will leave the company. I built a k-nearest neighbors (KNN) classifier and compared four feature sets to see how the choice of inputs affected churn predictions.

## Dataset

The dataset contains 667 customer records, with 15 potential predictors and one target column, `Churn`. The predictors include account length, phone usage, charges, and customer service calls. The target indicates whether a customer left the company.

Fewer customers left than stayed. That imbalance matters because a model can score well on overall accuracy while still missing many of the customers who churned.

## What I did

1. **Explored the data.** I used a correlation heatmap, boxplots, and a churn count plot to compare customer groups and examine patterns in the predictors.
2. **Prepared the features.** I encoded Boolean values and standardized the inputs so features with larger numeric values would have less influence on the model simply because of their scale.
3. **Compared four feature sets.** I tested all predictors, removed five weakly correlated features, tested five usage and service predictors, and evaluated a smaller four-feature model.
4. **Tuned and evaluated the model.** I used cross-validation to test different numbers of neighbors and reviewed confusion matrices, precision, recall, and F1 scores.

## What I learned

The confusion matrices showed two different kinds of errors. A false negative means missing a customer who leaves. A false positive means flagging a customer who stays. For a retention team, churn recall helps show how many departing customers the model identifies, while precision helps show how often its churn alerts are correct.

In the original analysis, some feature sets were evaluated on different test splits. My next step is to use one stratified train-test split for every comparison, fit scaling and tune the model within the training folds, and report churn recall and F1 on the held-out test set.

## Files

| File | What it contains |
| --- | --- |
| `customer_churn_prediction.py` | Data exploration, visualizations, model training, and evaluation |
| `churn.csv` | Telecom customer data used in the analysis |

## How to run

Download or clone the repository. Open a terminal in the project folder with both files, then run:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn
python customer_churn_prediction.py
```

The script prints model evaluation results and displays the charts.
