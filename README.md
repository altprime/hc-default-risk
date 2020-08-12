# Home Credit Default Risk

This dataset can be found [here](https://www.kaggle.com/c/home-credit-default-risk/data).

Often people find it challenging to acquire a loan due to insufficient credit history. [Home Credit](https://www.homecredit.net) strives to broaden financial inclusion for the unbanked people. In order to achieve their goals they hosted this competition on Kaggle whose primary objective is to predict whether an applicant will be able to repay their loan amount. This is a standard supervised learning (classification) problem.

### Data

Before we jump in, we need to understand the data that has been provided. There are various data sources:

- application_train/application_test: the main training and testing data with information about each loan application at Home Credit. Every loan has its own row and is identified by the feature `SK_ID_CURR`. The training application data comes with the `TARGET` indicating 0: the loan was repaid or 1: the loan was not repaid.
- bureau: data concerning client's previous credits from other financial institutions. Each previous credit has its own row in bureau, but one loan in the application data can have multiple previous credits.
- bureau_balance: monthly data about the previous credits in bureau. Each row is one month of a previous credit, and a single previous credit can have multiple rows, one for each month of the credit length.
- previous_application: previous applications for loans at Home Credit of clients who have loans in the application data. Each current loan in the application data can have multiple previous loans. Each previous application has one row and is identified by the feature `SK_ID_PREV`.
- POS_CASH_BALANCE: monthly data about previous point of sale or cash loans clients have had with Home Credit. Each row is one month of a previous point of sale or cash loan, and a single previous loan can have many rows.
- credit_card_balance: monthly data about previous credit cards clients have had with Home Credit. Each row is one month of a credit card balance, and a single credit card can have many rows.
- installments_payment: payment history for previous loans at Home Credit. There is one row for every made payment and one row for every missed payment.

The definitions of all features are provided in the `HomeCredit_columns_description.csv`.

We use `application_train.csv` and `application_test.csv` as our training and testing dataset respectively. Their shape is `(307511, 122)` and `(48744, 121)`. We can see that there is one column less in the test dataset. In order to find out which column is missing we rely on the `.difference()` function for dataframes. The results shows us that the `TARGET` column is not present in the test dataset.

### Exploratory Data Analysis

__Distribution of `TARGET`__

`TARGET` is what we need to predict. It's either a 0, indicating that the loan was paid on time or 1, indicating that there was a delay and/or difficulty in payment. We use the `.value_counts(normalize=True)` function to determine the number of people who fall in each category. This shows us that approximately 92% people were able to make timely payments, while 8% people faced some difficulties. The graph below serves as a pictorial representation of the same.

<p align="center">
  <img title='TARGET Distribution' src='https://github.com/anxrxdh/hc-default-risk/blob/master/plots/target-distribution.png'>
</p>

As we can see from the graph as well that there is a severe imbalance in the class and this will lead to the model being biased. 

__Missing Values__

Using a cascade of functions `.isna().any().sum()` we can see that of the 122 columns, 67 have missing values. While building a model these missing values will need taken care of.

> The imputation of missing values is carried out later, while building the baseline model

__Type of Columns Present__

It might be of immense help to how many columns belong to which data type. 

| dtype  | count |
| :----: | :---: |
| float  |  65   |
|  int   |  41   |
| object |  16   |

__Encoding Categorical Variables__

There are basically two ways of encoding categorical variabels:

1. One Hot Encoding: Every category is made into a new column and assigned either 0 or 1 based on whether it's value is True or not.
2. Label Encoding: Every category is assigned an integer number. The drawback of this being that the inherent property is not reflected by the number assigned since the assignment is random.

We'll use label encoding where number of categories is &le; 2 and one hot encoding where categories > 2. After encoding the train and test dataset, the shape was `(307511, 243)` and `(48744, 239)` respectively.

__Aligning the train and test datasets__

One-hot encoding has created more columns in the training data because there were some categorical variables with categories not represented in the testing data. To remove the columns in the training data that are not in the testing data, we need to `align` the dataframes. After completing this the shape of the train and test data is `307511, 240)` and `(48744, 239)` respectively. So, both datasets now have the same features which is exactly what we need.

__Outliers__

A quick glance at the output of the `.describe()` function shows that `DAYS_BIRTH` has negative values. This is because they're recorded relative to the current loan application. Hence we multiply and divide by `-1` and `365` respectively. Now, we can see that the range is reasonable and there aren't any outliers.

A similar conundrum was seen in `DAYS_EMPLOYED`, where the maximum value is about 1000 years. Let's look at the histogram for a better understanding.

<p align="center">
  <img title='Days Emp Histogram' src='https://github.com/anxrxdh/hc-default-risk/blob/master/plots/days-emp-histogram.png'>
</p>


Let's now subset the anomalous clients and see if they tend to have higher or low rates of default than the rest of the clients.

```python
The non-anomalies default on 8.66% of loans
The anomalies default on 5.40% of loans
There are 55374 anomalous days of employment
```

 It turns out that the anomalies have a lower rate of default. Since imputing anomalies is case to case procedure, in our case the best approach will be to fill in the anomalous values with not a number (`np.nan`) and then create a new Boolean column indicating whether or not the value was anomalous. The distribution looks something like this:

<p align="center">
  <img title='Days Emp Histogram - Anomalies' src='https://github.com/anxrxdh/hc-default-risk/blob/master/plots/days-hist-emp-anom.png'>
</p>

The distribution looks to be much more in line with what we would expect, and we also have created a new column to tell the model that these values were originally anomalous (because we will have to fill in the nans with some value, probably the median of the column). The other columns with `DAYS` in the dataframe look to be about what we expect with no obvious outliers.

As an extremely important note, anything we do to the training data we also have to do to the testing data. Let's make sure to create the new column and fill in the existing column with `np.nan` in the testing data.

__Correlations__

The `DAYS_BIRTH` has the highest positive correlation. Revisiting the documentation, `DAYS_BIRTH` is the age in days of the client at the time of the loan in negative days. The correlation is positive, but the value of this feature is actually negative, meaning that as the client gets older, they are less likely to default on their loan (target == 0). That's a little confusing, so we will take the absolute value of the feature and then the correlation will be negative. This value turn out to be -0.078.

_Effect of Age_: As the client gets older, there is a negative linear relationship with the target meaning that as clients get older, they tend to repay their loans on time more often.

### Feature Engineering

As we can see the clear class imbalance in the target variable and glaring discrepancies in various other variables, it is not hard to guess that a lot of feature engineering will be required. We'll stick to two fundamental feature construction methods

1. Polynomial Features
2. Domain Features

__Polynomial Features__

Here, we'll make features that are powers of existing features as well as interaction terms (features that are a combination of multiple individual features) between existing features. Apart from making a new dataframe for polynomial features, we'll also impute the missing data we had encountered before.

We'll create polynomial features using the `EXT_SOURCE` variables and the `DAYS_BIRTH` variable. After creating these new polynomials we see that we have considerable features. To get the names we have to use the polynomial features `.get_feature_names()` method. We see that there are 35 features with individual features raised to powers up to degree 3 and interaction terms. Now, we can see whether any of these new features are correlated with the target.

```python
EXT_SOURCE_2 EXT_SOURCE_3                -0.193939
EXT_SOURCE_1 EXT_SOURCE_2 EXT_SOURCE_3   -0.189605
EXT_SOURCE_2 EXT_SOURCE_3 DAYS_BIRTH     -0.181283
EXT_SOURCE_2^2 EXT_SOURCE_3              -0.176428
EXT_SOURCE_2 EXT_SOURCE_3^2              -0.172282
EXT_SOURCE_1 EXT_SOURCE_2                -0.166625
EXT_SOURCE_1 EXT_SOURCE_3                -0.164065
EXT_SOURCE_2                             -0.160295
EXT_SOURCE_2 DAYS_BIRTH                  -0.156873
EXT_SOURCE_1 EXT_SOURCE_2^2              -0.156867
Name: TARGET, dtype: float64
DAYS_BIRTH     -0.078239
DAYS_BIRTH^2   -0.076672
DAYS_BIRTH^3   -0.074273
TARGET          1.000000
1                    NaN
Name: TARGET, dtype: float64
```

We see that several features have a greater correlation with the target as compared to the original features. It will be helpful, while building a model, to see whether or not these have any impact. In order to evaluate the various models, we'll add these features to a copy of the original dataset.

__Domain Features__

These are features that are specific to the domain, in our case that is default risk (broadly, finance). Here we can make features we think have a significant impact on the credit-worthiness of a client. some feature that we'll focus on here are:

- `CREDIT_INCOME_PERCENT`: the percentage of the credit amount relative to a client's income
- `ANNUITY_INCOME_PERCENT`: the percentage of the loan annuity relative to a client's income
- `CREDIT_TERM`: the length of the payment in months (since the annuity is the monthly amount due
- `DAYS_EMPLOYED_PERCENT`: the percentage of the days employed relative to the client's age

__Visualizing the new domain features__

<p align="center">
  <img title='Domain Features' src='https://github.com/anxrxdh/hc-default-risk/blob/master/plots/domain-features-viz.png'>
</p>

### Predictive Models

Let's now dive into using all these features we have created. We'll start with a baseline model, which will essentially be a barebones model. From there we'll build a more complex model.

__Baseline Logistic Regression__

We'll develop the baseline model by including all the encoded variables. We'll then preprocess that data by _imputing_ missing values and normalizing the range of features.

For our model, we'll keep `C` (the regularization parameter) quite low. This will allow the model to prevent overfitting albeit at the cost of adding bias. In consequence, we'll get a better result compared to the default parameter setting of `LogisticRegression()` but it will still be 'baseline' and probably won't stand competition to future models.

After training the model, we'll predict the model's performance over the testing set. It may be noted that since we're predicting probabilities, we'll be using `.predict_proba()` function. This will return an array of shape [n_observations, 2]. The first column is the probability of the target being 0 and the second column is the probability of the target being 1 (so for a single row, the two columns must sum to 1). We want the probability the loan is not repaid, so we will select the second column.

Additionally, we want our results to follow a specific format. We're only concerned with two columns: `SK_ID_CURR` and `TARGET`. The final output looks something like this:

<p align="center">
  <img title='Logistic Regression Prediction' src='https://github.com/anxrxdh/hc-default-risk/blob/master/plots/log-reg-pred.JPG'>
</p>


> It may be noted that the test dataset came without the `TARGET` variable since this was a Kaggle hosted competition. Since we're unaware of the target values in the testing set, it will be impossible to test the accuracy of our model. The evaluation parameters mentioned in the documentation provided are ROC AUC characteristics.

__Random Forest__

Next we jump into random forests in an attempt to improve our model performance. The procedure is the same so far. Additionally, we'll also extract _feature importance_ of the various variables. This gives us an idea about each feature's impact on the model. We'll plot these at a later stage. The predictions of the random forest model look something like this:

<p align="center">
  <img title='Random Forest Prediction' src='https://github.com/anxrxdh/hc-default-risk/blob/master/plots/rf-pred.JPG'>
</p>


__Random Forest: Polynomial Features__

In the previous section we looked at the random forest model on default features. It's time to consider the impact of our engineered features. Here's a look at the predictions:

<p align="center">
  <img title='Random Forest: Polynomial Features' src='https://github.com/anxrxdh/hc-default-risk/blob/master/plots/poly-pred.JPG'>
</p>


__Plotting Feature Importance__

In order to plot the extracted features, we'll simply write a function that will show us the top 15 features. Let's us look at the graph:

<p align="center">
  <img title='Feature Importance' src='https://github.com/anxrxdh/hc-default-risk/blob/master/plots/feat-imp-default.png'>
</p>


### Future Scope

You can customize the function to plot the bottom 15 and remove them. Reducing the number of feature does help in preventing overfitting and consequently drawing unwarranted conclusions. Although, as per my experience, this improvement in the model will be marginal at best.

You can also plot feature importance of polynomial and domain features that we engineered and use the top variables from all three plot to develop a more robust model. This model should be able to perform much better.
