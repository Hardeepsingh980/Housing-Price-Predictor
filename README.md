# Housing-Price-Predictor
This is an simple machine learning program where i have used some models and data to predict the price of house in a particular area of Boston. The data used was collected in Boston.

# Usage 
Run the Model_trainer.py file to first create a Model.

Now You can use Model_usage.py file to predict the price in Dollars($) of the house on the behalf of given data.

To Check The Features Included In Data Refer To features.txt File. All the features has been described

(The Prediction made were based on the data collected in Boston a decade before. So, the predictions may not make sense today. It is for demonstration purpose only.)

# Output 

The Outputs I got From Using Different Models on The Same Data :-

1. 

	Model Used :- LinearRegression

	Mean: 5.057284756294412

	Standard Deviation: 1.033541177118166 

2.

	Model Used :- DecisionTreeRegressor

	Mean: 4.612070280128845

	Standard Deviation: 1.3159754054455628 

3.

	Model Used :- RandomForestRegressor

	Mean: 3.6059654978067237

	Standard Deviation: 1.0097245063023408 

4.

	Model Used :- Ridge

	Mean: 5.054315161372322

	Standard Deviation: 1.039322247483624 

5.

	Model Used :- RidgeCV

	Mean: 5.035925789578547

	Standard Deviation: 1.085382992419537 

6.

	Model Used :- Lasso

	Mean: 5.409101751828857

	Standard Deviation: 1.3605785740655751 

7.

	Model Used :- BasyesianRidge

	Mean: 5.042783866909807

	Standard Deviation: 1.0733113572540294 


Best Performing Model Was :- RandomForestRegressor
