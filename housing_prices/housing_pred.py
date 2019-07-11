#%%
import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#%%
# Read in downloaded data
FILE_LOCATION = os.path.join(os.getcwd(), 'machine_learning_orielly', 'housing_prices', 'datasets')
def load_housing_data(housing_path=FILE_LOCATION):
    '''
    Load housing pricing dataset from datasets directory
    '''
    csv_path = os.path.join(FILE_LOCATION, 'housing.csv')
    return pd.read_csv(csv_path)

#%%
housing = load_housing_data()
housing.head()

#%%
housing.info()

#%%
# Lets view the value count totals of the Ocean Proximity column
# of the dataset
housing.ocean_proximity.value_counts()

#%%
housing.describe()

#%%
housing.hist(bins=50, figsize=(20,15))
plt.show()

#%%
# Creating test set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

#%%
print(len(train_set))
print(len(test_set))

#%%
housing['income_cat'] = pd.cut(x=housing['median_income'],bins=[0., 1.5, 3., 4.5, 6., np.inf], 
                                labels= [1,2,3,4,5])

housing.income_cat.hist()

#%%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#%%
# Lets vien the income category proportions in th test set.
strat_test_set['income_cat'].value_counts() / len(strat_test_set)

#%%
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

#%%
# EDA
housing = strat_train_set.copy()

#%%
# Scatter geographical info
housing.plot(kind='scatter', x='longitude', 
                            y='latitude', 
                            alpha=0.4, 
                            s=housing['population']/100, 
                            label='population',
                            figsize=(10,7),
                            c='median_house_value',
                            cmap=plt.get_cmap('jet'),
                            colorbar=True
                            )
plt.legend()

#%%
# Lets check correlation coef's for Median House Values
corr_matrix = housing.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)

#%%
from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize=(12,8))

#%%
# Focus on median_income and median_house_vale
housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)

#%%
housing['rooms_per_household'] = housing['total_rooms']/ housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']

corr_matrix = housing.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)

#%%
# Lets fill in the missing values of the numeric data with feature median.
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()


num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), 
                        ('std_scaler', StandardScaler()),
                        ])

housing_num = housing.drop('ocean_proximity', axis=1)

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipline = ColumnTransformer([('num', num_pipeline, num_attribs),
                                    ('cat', OneHotEncoder(), cat_attribs)
                                    ])

housing_prepared = full_pipline.fit_transform(housing)

#%%
# Building a Regression Model
from sklearn.linear_model import LinearRegression
# housing_prepared = pd.concat([housing_tr, housing_cat_tr], axis=1)
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#%%
# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipline.fit_transform(housing.iloc[:5])

print('Predictions: ', lin_reg.predict(housing_prepared[:5]))
print('Labels: \n', housing_labels.iloc[:5], sep='')

#%%
# Let check our prediction errors of the linear regression model
from sklearn.metrics import mean_squared_error

y_pred_lin = lin_reg.predict(housing_prepared[:5])
y_true = housing_labels.iloc[:5]
mse_lin = mean_squared_error(y_true, y_pred_lin)

#%%
# I will compare the result from the linear model with those from a more powerful
# Decision tree model

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor()
tree.fit(housing_prepared, housing_labels)

#%%
# Predicting use the tree based model
y_pred_tree = tree.predict(housing_prepared)
y_true = housing_labels

mse_tree = mean_squared_error(y_true, y_pred_tree)

#%%
# I will compare the result from the linear model with those from a more powerful
# Random Forest

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(housing_prepared, housing_labels)


#%%
# Predicting use the Random Forest model
y_pred_rfr = rfr.predict(housing_prepared)
y_true = housing_labels

mse_rfr = mean_squared_error(y_true, y_pred_rfr)


#%%
# Displaying cross validation scores

def display_scores(model_name, model, mse):
    from sklearn.model_selection import cross_val_score
    score = cross_val_score(model, 
                        housing_prepared, 
                        housing_labels, 
                        scoring='neg_mean_squared_error',
                        cv=10)

    rmse = np.sqrt(-score)
    print(model_name)
    print('Training set error:', np.sqrt(mse))
    print('Cross val score errors:', rmse)
    print('Mean of Scores:', np.mean(rmse))
    print('Std of Scores:', np.std(rmse))
    print()


display_scores(model_name='Linear Regression', model=lin_reg, mse=mse_lin)
display_scores(model_name='Decsion Tree Regression', model=tree, mse=mse_tree)
display_scores(model_name='Random Forest Regression', model=rfr, mse=mse_rfr)


#%%
# I will compare the result from the linear model with those from a more powerful
# Random Forest with Grid Search

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Grid search param grid
param_grid = [
    {'n_estimators' : [30, 50, 70],
    'max_features' : [6, 8, 10, 12]
    },
    {'n_estimators' : [30, 50],
    'max_features' : [6, 8, 10, 12]
    },
]

rfr = RandomForestRegressor()
# rfr.fit(housing_prepared, housing_labels)
grid_search = GridSearchCV(rfr, 
                            param_grid, 
                            cv=5, 
                            scoring='neg_mean_squared_error',
                            return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
#%%
print(grid_search.best_params_)
#%%
print(grid_search.best_estimator_)
#%%
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

#%%
# Trying out the test set on the tuned model
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('Final rmse of tuned model:', final_rmse)

#%%
# Of this California housing prices example data,
# 'the median income' of homes was the best predictor of houseing price.
