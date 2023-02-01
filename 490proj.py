# -*- coding: utf-8 -*-
"""
Created on Mon April 26 15:53:24 2021

@author: Hasan Ramadan
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from xgboost import XGBRegressor
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

def refine(y1_pred):
    for i in range(len(y1_pred)):
        if y1_pred[i]>100:
            y1_pred[i] = 100
        y1_pred[i] = round(y1_pred[i])
    return y1_pred
        
#Preparing and preprocessing the data        
df = pd.read_csv("m3p.csv")
df = pd.get_dummies(df, drop_first=True)
df.replace(np.nan,0,inplace = True)


X = np.array(df.drop(['STRESS/100', 'WorkLoad/100'], 1))
y1 = np.array(df['WorkLoad/100'])
y2 = np.array(df['STRESS/100'])

X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size = 0.2, random_state = 0)
X_train, X_test, y2_train, y2_test = train_test_split(X, y2, test_size = 0.2, random_state = 0)





#Linear Regression Model
lr1 = LinearRegression()
lr2 = LinearRegression()

lr1.fit(X_train, y1_train)
lr2.fit(X_train, y2_train)

y1_pred = refine(lr1.predict(X_test))
y2_pred = refine(lr2.predict(X_test))

r2_1 = round(r2_score(y1_test,y1_pred),2) #for Workload
r2_2 = round(r2_score(y2_test,y2_pred),2) #for Stress

mse_1 = round(mean_squared_error(y1_test,y1_pred),2) #for Workload
mse_2 = round(mean_squared_error(y2_test,y2_pred),2) #for Stress

print("THE R^2 SCORE for linear regression     " + str((r2_1,r2_2)))
print("THE MSE for linear regression           " + str((mse_1,mse_2)))
print(" ")





#Ridge Regression (Regularization)
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)
    std_error = cv_scores_std / np.sqrt(10)
    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []
# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)
# Compute scores over range of alphas
for alpha in alpha_space:
    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha  
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X_train, y1_train, cv=10)  
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))  
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))
# Display the plot
display_plot(ridge_scores, ridge_scores_std)





#Decision Tree Model
dt1 = DecisionTreeRegressor()
dt2 = DecisionTreeRegressor()

dt1.fit(X_train,y1_train)
dt2.fit(X_train,y2_train)

y1_pred = refine(dt1.predict(X_test))
y2_pred = refine(dt1.predict(X_test))

r2_1 = round(r2_score(y1_test,y1_pred),2) #for Workload
r2_2 = round(r2_score(y2_test,y2_pred),2) #for Stress

mse_1 = round(mean_squared_error(y1_test,y1_pred),2) #for Workload
mse_2 = round(mean_squared_error(y2_test,y2_pred),2) #for Stress

print("THE R^2 SCORE for DT                    " + str((r2_1,r2_2)))
print("THE MSE for DT                          " + str((mse_1,mse_2)))
print(" ")





#Extra Tree Model
reg1 = ExtraTreeRegressor()
reg2 = ExtraTreeRegressor()

reg1.fit(X_train,y1_train)
reg2.fit(X_train,y2_train)

y1_pred = refine(reg1.predict(X_test))
y2_pred = refine(reg1.predict(X_test))

r2_1 = round(r2_score(y1_test,y1_pred),2) #for Workload
r2_2 = round(r2_score(y2_test,y2_pred),2) #for Stress

mse_1 = round(mean_squared_error(y1_test,y1_pred),2) #for Workload
mse_2 = round(mean_squared_error(y2_test,y2_pred),2) #for Stress

print("THE R^2 SCORE for ExtraTree             " + str((r2_1,r2_2)))
print("THE MSE for ExtraTree                   " + str((mse_1,mse_2)))
print(" ")





#Neural Network
NN_model1 = Sequential()
NN_model2 = Sequential()
# The Input Layer :
NN_model1.add(Dense(68, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))
NN_model2.add(Dense(68, kernel_initializer='normal',input_dim = X.shape[1], activation='relu'))

# The Hidden Layers :
NN_model1.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model1.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model1.add(Dense(256, kernel_initializer='normal',activation='relu'))

NN_model2.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model2.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model2.add(Dense(256, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model1.add(Dense(1, kernel_initializer='normal',activation='linear'))
NN_model2.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model1.fit(X_train, y1_train, epochs=50, batch_size = 50, validation_split = 0.2, verbose = 0)

NN_model2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model2.fit(X_train, y2_train, epochs=50, batch_size = 50, validation_split = 0.2, verbose = 0)

y1_pred = NN_model1.predict(X_test)
y2_pred = NN_model2.predict(X_test)

r2_1 = round(r2_score(y1_test,y1_pred),2) #for Workload
r2_2 = round(r2_score(y2_test,y2_pred),2) #for Stress

mse_1 = round(mean_squared_error(y1_test,y1_pred),2) #for Workload
mse_2 = round(mean_squared_error(y2_test,y2_pred),2) #for Stress

print("THE R^2 SCORE for Nueral Network        " + str((r2_1,r2_2)))
print("THE MSE for Neural Network              " + str((mse_1,mse_2)))
print(" ")





#XGB Model
XGBModel1 = XGBRegressor()
XGBModel1.fit(X_train,y1_train , verbose=False)

XGBModel2 = XGBRegressor()
XGBModel2.fit(X_train,y2_train , verbose=False)

y1_pred = refine(XGBModel1.predict(X_test))
y2_pred = refine(XGBModel2.predict(X_test))


r2_1 = round(r2_score(y1_test,y1_pred),2) #for Workload
r2_2 = round(r2_score(y2_test,y2_pred),2) #for Stress

mse_1 = round(mean_squared_error(y1_test,y1_pred),2) #for Workload
mse_2 = round(mean_squared_error(y2_test,y2_pred),2) #for Stress

print("THE R^2 SCORE for ExtraGradientBoosting " + str((r2_1,r2_2)))
print("THE MSE for ExtraGradientBoosting       " + str((mse_1,mse_2)))




















