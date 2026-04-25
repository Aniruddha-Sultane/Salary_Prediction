import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r"C:\Users\Aniruddha\Downloads\Salary_Data (1).csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience(test set)')
plt.ylabel('Salary')
plt.show()


m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)


y_12 = m * 12 + c
print(y_12)

y_20 = m * 20 + c
print(y_20)

y_10 = m * 10 + c
print(y_10)

y_28 = m*28+c
print(y_28)


bias = regressor.score(x_train, y_train)
bias

variance = regressor.score(x_test, y_test)
variance


#stats for ml

dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset.var()

dataset.std()


from scipy.stats import variation
variation(dataset.values)

variation(dataset['Salary'])


dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

dataset.skew()


dataset['Salary'].skew()


dataset.sem()


dataset['Salary'].sem()


import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])



a =dataset.shape[0]
b = dataset.shape[1]

degree_of_freedom = a-b
print(degree_of_freedom)


#ssr
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#sse
y=y[0:6]
SSE = np.sum((y_pred-y_mean)**2)
print(SSE)

#sst
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)


#r2
r_square = 1-SSR/SST
print(r_square)



import pickle
filename = 'Simple_linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("model has been picked and saved as Simple_linear_regression_model.pkl")

import os 
print(os.getcwd())