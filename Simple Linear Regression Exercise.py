import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
df=pd.read_csv(url)

#Verify a successful load with randomly selected records
df.sample(5)

df.describe()

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.sample(9)

viz = cdf[['CYLINDERS', 'ENGINESIZE', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
plt.show()

# #Fuel Consumption and Emissions relationship
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel('Emission')
plt.show()

#Engine Size and Emission relationship
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('Engine size')
plt.ylabel('Emission')
plt.xlim(0,27)
plt.show()

#Plotting Cylinger against CO2 Emission 
#Exercise 1
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('CYLINDERS')
plt.ylabel('Emission')
plt.show()

x = cdf.ENGINESIZE.to_numpy()
y = cdf.CO2EMISSIONS.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#Building a simple linear regression model 

#Creating a model object 
regressor = linear_model.LinearRegression()

#train the model on the training data
# X_train is a 1-D array but sklearn models expect a 2D array as input for the training data, with shape (n_observations, n_features).
# So we need to reshape it. We can let it infer the number of observations using '-1'.
regressor.fit(x_train.reshape(-1, 1), y_train)

#Print the coefficients
print('Coefficients: ', regressor.coef_[0]) # with simple linear regression there is only one coefficient, here we extract it from the 1 by 1 array.
print('Intercept: ', regressor.intercept_)

#Coefficient and Intercept are regression parameters determined by the model.
#They define the slope and intercept of the 'best-fit' line to the training data

#Visualizing the goodness-of-fit 
#The regression model is the line given by y = intercept + coeffcient * x
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, regressor.coef_ * x_train + regressor.intercept_, '-r')

#Use the predict method to make test predicitions
y_test_ = regressor.predict(x_test.reshape(-1,1))

#Evaluation 
print("Mean Absolute Error: %.2f" % mean_absolute_error(y_test, y_test_))
print("Mean Squared Error: %.2f" % mean_squared_error(y_test, y_test_))
print("Root Mean Squared Error %.2f" % np.sqrt(mean_squared_error(y_test, y_test_)))
print("R2-score: %.2f" % r2_score(y_test, y_test_))


#Practice

#1.Plot the regression model result over the test data instead of the training data. Visually evaluate whether the result is good.
plt.scatter(x_test, y_test, color='blue')
plt.plot(x_test, regressor.coef_ * x_test + regressor.intercept_, "-r")
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#2 Select the fuel consumption feature from the dataframe and split the data 80%/20% into training and testing sets.¶

x = cdf.FUELCONSUMPTION_COMB.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#3 Train a linear regression model using the training data you created.

regr = linear_model.LinearRegression()

regr.fit(x_train.reshape(-1,1), y_train)

#4 Use the model to make test predictions on the fuel consumption testing data.
y_test_ = regr.predict(x_test.reshape(-1,1))

#5 Calculate and print the Mean Squared Error of the test predictions.
print("Mean Squared Error %2f" % mean_squared_error(y_test, y_test_))
