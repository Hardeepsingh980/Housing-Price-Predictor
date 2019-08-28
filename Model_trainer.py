import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

# Output Model
from joblib import dump, load



# Loading The Data Frame

housing = pd.read_csv('log.csv')
housing.drop('num', axis = 1,inplace=True)

print(housing.head())

print('\n\n\n')




# Train Test Splitting

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)





# checking for important perameters

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]



# Splitting data for training purpose

housing = strat_train_set.copy()




# Looking For Co-relations

corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


attributes = ['MEDV','RM','ZN','LSTAT']
scatter_matrix(housing[attributes], figsize=(12,8))


housing.plot(kind='scatter', x='RM',y='MEDV',alpha=0.8)



# Attribute Combination

housing['TEXRM'] = housing['TAX']/housing['RM']




corr_matrix = housing.corr()
corr_matrix['TEXRM'].sort_values(ascending=False)

housing.plot(kind='scatter', x='TEXRM',y='MEDV',alpha=0.8)

housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set['MEDV'].copy()





# Creating Pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler())
    # Add as Many as you want
])


housing_num_tr = my_pipeline.fit_transform(housing)





# Loop For All The Models

models = [LinearRegression, DecisionTreeRegressor, RandomForestRegressor, linear_model.Ridge, linear_model.RidgeCV, linear_model.Lasso,linear_model.BayesianRidge]

name_models = ['LinearRegression', 'DecisionTreeRegressor','RandomForestRegressor','Ridge','RidgeCV','Lasso','BasyesianRidge']

result = f'\t\tModels Output\n\t\t{"="*len("Models Output")} \n\nOutputs From Different Models Are Displayed Here :- \n\n\n'

mean_list = []


for i, i_name in zip(models, name_models):
        
    model = i()

    model.fit(housing_num_tr,housing_labels)

    some_data = housing.iloc[:5]

    some_labels = housing_labels.iloc[:5]

    prepared_data = my_pipeline.transform(some_data)

    model.predict(prepared_data)

    housing_predictions = model.predict(housing_num_tr)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = np.sqrt(mse)




    # Using Better Evaluation Technique - Cross Validation

    scores = cross_val_score(model, housing_num_tr, housing_labels,scoring='neg_mean_squared_error'
                             , cv=10)

    rmse_scores = np.sqrt(-scores)



    def print_score(score):

        global result
        result +=  f'-->  Model Used :- {i_name}\n\n\tMean: {rmse_scores.mean()}\n\n\tStandard Deviation: {rmse_scores.std()} \n\n\n'
        mean_list.append(rmse_scores.mean())

    print_score(rmse_scores)
    


result += 'Best Performing Model Was :- '+ str(name_models[mean_list.index(min(mean_list))])


print('\n\n\n')
print(result)



f = open('Result_output.txt','w')

f.write(result)

f.close()

print('\n\n\n Results Has Been Saved In The "Result_output.txt" In Same Directory.')










# Choosing the Best Models and Performing Further

model = models[mean_list.index(min(mean_list))]()

model.fit(housing_num_tr,housing_labels)



print('\n\n\n\n')


# Testing The Model on test data

x_test = strat_test_set.drop('MEDV', axis=1)
y_test = strat_test_set['MEDV'].copy()

x_test_prepared = my_pipeline.transform(x_test)

final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print('Testing The Model On test data :-\npredicted value : orignal value')

for i, i_y in zip(final_predictions[:10], list(y_test)[:10]):
      print('\t',int(i),' : ', int(i_y), end='\n')




# Saving The Model

dump(model, 'Dragon.Joblib')


# Using The Model

model = load('Dragon.joblib')


features = np.array([[0.02, 70,10,1,0.5380,3,20,4,1,0,30,400,30]])

print(model.predict(features))



