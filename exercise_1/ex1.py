#!/usr/bin/env python

"""
Machine Learning Online Class - Exercise 1: Linear Regression
Cloned this from https://github.com/zduey/machinelearning_coursera
Made some modification
Modified by github.com/utkarshmani1997
"""

# Initialization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ex1_utils import * # imports all functions in ex1_utils.py

# ======================= Plotting =======================
print('Plotting Data ...\n')
data = pd.read_csv("ex1data1.txt",names=["X","y"])
x = np.array(data.X)[:,None] # population in 10,0000
y = np.array(data.y) # profit for a food truck
m = len(y) # number of training examples

# Plot Data
fig = plotData(x,y)
fig.show()

input('Program paused. Press enter to continue.\n')

## =================== Gradient descent ===================
print('Running Gradient Descent ...\n')

ones = np.ones_like(x)
X = np.hstack((ones,x)) # Add a column of ones to x
theta = np.zeros(2) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# compute and display initial cost
computeCost(X, y, theta)

# run gradient descent
theta, hist = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: ')
print(theta[0],"\n", theta[1])

# Plot the linear fit
plt.plot(x,y,'rx',x,np.dot(X,theta),'b-')
plt.legend(['Training Data','Linear Regression'])
plt.show()

a=input("Program paused. Enter the population:")
b=int(a)/10000;

# Predict values for the given population
predict1 = np.dot([1, b],theta) # takes inner product to get y_bar
print('For population = {} , we predict a profit of '.format(b), predict1*10000)
