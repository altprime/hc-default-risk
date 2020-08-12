import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns 

from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer

print(os.listdir('./data/'))

# reading training data
train = pd.read_csv('./data/application_train.csv')
test = pd.read_csv('./data/application_test.csv')
train.shape
test.shape

# check which column is missing in the test dataset
train.columns.difference(test.columns)

# distribution of TARGET 
train['TARGET'].value_counts(normalize=True)
train['TARGET'].astype(int).plot.hist();

# check missing values
train.isna().any().sum()
sns.heatmap(train.isna(), cbar=False)

# column types
train.dtypes.value_counts()

# unique classes in each categorical variable
train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

# encoding categorical varaibles
## label encoding
train_copy = train.copy()
test_copy = test.copy()
lenc = LabelEncoder()
lenc_count = 0
# iterate through cols
for col in train_copy:
    if train_copy[col].dtype == 'object':
        # if 2 or fewer unique categories
        if len(list(train_copy[col].unique())) <= 2:
            # train on the training data
            lenc.fit(train_copy[col])
            # Transform both training and testing data
            train_copy[col] = lenc.transform(train_copy[col])
            test_copy[col] = lenc.transform(test_copy[col])
            
            # Keep track of how many columns were label encoded
            lenc_count += 1
            
print('%d columns were label encoded.' % lenc_count)
## one hot encoding
train_copy = pd.get_dummies(train_copy)
test_copy = pd.get_dummies(test_copy)
print('Training Features shape: ', train_copy.shape)
print('Testing Features shape: ', test_copy.shape)

# aligning the datasets
train_labels = train_copy['TARGET']
# keep only columns present in both dataframes
train_copy, test_copy = train_copy.align(test_copy, join = 'inner', axis = 1)
# add the target back in
train_copy['TARGET'] = train_labels
print('Training Features shape: ', train_copy.shape)
print('Testing Features shape: ', test_copy.shape)

# outliers
## DAYS_BIRTH
(train_copy['DAYS_BIRTH'] / -365).describe()
## DAYS_EMPLOYED
train_copy['DAYS_EMPLOYED'].describe()
train_copy['DAYS_EMPLOYED'].plot.hist(title = 'Days Employed Histogram');
plt.xlabel('Days Employed');

anom = train_copy[train_copy['DAYS_EMPLOYED'] == 365243]
non_anom = train_copy[train_copy['DAYS_EMPLOYED'] != 365243]
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))
print('There are %d anomalous days of employment' % len(anom))

# create an anomalous flag column
train_copy['DAYS_EMPLOYED_ANOM'] = train_copy["DAYS_EMPLOYED"] == 365243
# replace the anomalous values with nan
train_copy['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
train_copy['DAYS_EMPLOYED'].plot.hist(title = 'Days Employed Histogram');
plt.xlabel('Days Employed');

# same for test
test_copy['DAYS_EMPLOYED_ANOM'] = test_copy["DAYS_EMPLOYED"] == 365243
test_copy["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)
print('There are %d anomalies in the test data out of %d entries' % (test_copy["DAYS_EMPLOYED_ANOM"].sum(), len(test_copy)))

# correlations
correlations = train_copy.corr()['TARGET'].sort_values()
# display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))

# effect of age
train_copy['DAYS_BIRTH'] = abs(train_copy['DAYS_BIRTH'])
train_copy['DAYS_BIRTH'].corr(train_copy['TARGET'])


'''
FEATURE ENGINEERING
'''

# feature engineering
# make df for polynomial features
poly_features = train_copy[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
poly_features_test = test_copy[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
# impute missing values
imputer = SimpleImputer(strategy = 'median')
poly_target = poly_features['TARGET']
poly_features = poly_features.drop(columns = ['TARGET'])
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)
# create polynomial features
poly_transformer = PolynomialFeatures(degree = 3)
poly_transformer.fit(poly_features)
# transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)

poly_transformer.get_feature_names(input_features = ['EXT_SOURCE_1', 
                                                     'EXT_SOURCE_2', 
                                                     'EXT_SOURCE_3', 
                                                     'DAYS_BIRTH'])[:15]

# create df for features
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 
                                                                           'EXT_SOURCE_2',
                                                                           'EXT_SOURCE_3', 
                                                                           'DAYS_BIRTH']))
# add in the target
poly_features['TARGET'] = poly_target
# find the correlations with the target
poly_corrs = poly_features.corr()['TARGET'].sort_values()
# display most negative and most positive
print(poly_corrs.head(10))
print(poly_corrs.tail(5))

# add test features into dataframe
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 
                                                                                'EXT_SOURCE_2', 
                                                                                'EXT_SOURCE_3', 
                                                                                'DAYS_BIRTH']))

# merge polynomial features into training dataframe
poly_features['SK_ID_CURR'] = train_copy['SK_ID_CURR']
train_copy_poly = train_copy.merge(poly_features, on = 'SK_ID_CURR', how = 'left')

# add polnomial features into testing dataframe
poly_features_test['SK_ID_CURR'] = test_copy['SK_ID_CURR']
test_copy_poly = test_copy.merge(poly_features_test, on = 'SK_ID_CURR', how = 'left')

# Align the dataframes
train_copy_poly, test_copy_poly = train_copy_poly.align(test_copy_poly, join = 'inner', axis = 1)

# Print out the new shapes
print('Training: polynomial features shape: ', train_copy_poly.shape)
print('Testing: polynomial features shape:  ', test_copy_poly.shape)



# domain features
train_domain = train.copy()
test_domain = test.copy()

train_domain['CREDIT_INCOME_PERCENT'] = train_domain['AMT_CREDIT'] / train_domain['AMT_INCOME_TOTAL']
train_domain['ANNUITY_INCOME_PERCENT'] = train_domain['AMT_ANNUITY'] / train_domain['AMT_INCOME_TOTAL']
train_domain['CREDIT_TERM'] = train_domain['AMT_ANNUITY'] / train_domain['AMT_CREDIT']
train_domain['DAYS_EMPLOYED_PERCENT'] = train_domain['DAYS_EMPLOYED'] / train_domain['DAYS_BIRTH']

test_domain['CREDIT_INCOME_PERCENT'] = test_domain['AMT_CREDIT'] / test_domain['AMT_INCOME_TOTAL']
test_domain['ANNUITY_INCOME_PERCENT'] = test_domain['AMT_ANNUITY'] / test_domain['AMT_INCOME_TOTAL']
test_domain['CREDIT_TERM'] = test_domain['AMT_ANNUITY'] / test_domain['AMT_CREDIT']
test_domain['DAYS_EMPLOYED_PERCENT'] = test_domain['DAYS_EMPLOYED'] / test_domain['DAYS_BIRTH']


plt.figure(figsize = (12, 20))
# iterate through the new features
for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
    
    # create a new subplot for each source
    plt.subplot(4, 1, i + 1)
    # plot repaid loans
    sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 0, feature], label = 'target == 0')
    # plot loans that were not repaid
    sns.kdeplot(train_domain.loc[train_domain['TARGET'] == 1, feature], label = 'target == 1')
    
    # Label the plots
    plt.title('Distribution of %s by Target Value' % feature)
    plt.xlabel('%s' % feature); plt.ylabel('Density');
    
plt.tight_layout(h_pad = 2.5)



'''
BASELINE MODEL
'''

# logistic regression
from sklearn.preprocessing import MinMaxScaler

# drop the target from the training data
if 'TARGET' in train:
    training = train_copy.drop(columns = ['TARGET'])
else:
    training = train_copy.copy()
# feature names
features = list(training.columns)
# copy of testing
testing = test_copy.copy()
# median imputation of missing values
imputer = SimpleImputer(strategy = 'median')
# scale all feature between 0 and 1
scaler = MinMaxScaler(feature_range = (0, 1))
# fit on training data
imputer.fit(training)
# transform both training and testing data
training = imputer.transform(training)
testing = imputer.transform(test_copy)
# Repeat with the scaler
scaler.fit(training)
training = scaler.transform(training)
testing = scaler.transform(testing)

print('Training data shape: ', training.shape)
print('Testing data shape: ', testing.shape)




# model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 0.0001)
# train model
log_reg.fit(training, train_labels)
# predictions
# select the second column only
log_reg_pred = log_reg.predict_proba(testing)[:, 1]
# final dataframe
submit = test[['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred
submit.head()




# improved: random forest
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 100, 
                                       random_state = 50, 
                                       verbose = 1, 
                                       n_jobs = -1)
# train model
random_forest.fit(training, train_labels)
# extract feature importances
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
# predictions
predictions = random_forest.predict_proba(testing)[:, 1]
#final dataframe
submit = test[['SK_ID_CURR']]
submit['TARGET'] = predictions
submit.head()



# random forest: polynomial features
poly_features_names = list(train_copy_poly.columns)

# Impute the polynomial features
imputer = SimpleImputer(strategy = 'median')

poly_features = imputer.fit_transform(train_copy_poly)
poly_features_test = imputer.transform(test_copy_poly)

# scale the polynomial features
scaler = MinMaxScaler(feature_range = (0, 1))
poly_features = scaler.fit_transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)

random_forest_poly = RandomForestClassifier(n_estimators = 100, 
                                            random_state = 50, 
                                            verbose = 1, 
                                            n_jobs = -1)
# train model
random_forest_poly.fit(poly_features, train_labels)
# feature importance
feature_importance_values_poly = random_forest_poly.feature_importances_
feature_importances_poly = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
# predictions
# predictions
predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]
submit = test[['SK_ID_CURR']]
submit['TARGET'] = predictions
submit.head()





# plot feature importance
def plot_feature_importances(df):
        
    # sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # normalize feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot() 
    # reverse index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('normalized importance'); plt.title('feature importance')
    plt.show()
    
    return df

# default features
plot_feature_importances(feature_importances)




















