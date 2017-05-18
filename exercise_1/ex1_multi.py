#!/usr/bin/env python

"""
Machine Learning Online Class Coursera
Exercise 1: Linear regression with multiple variables
Modified with few changes,Solved the issue
Modified by github.com/utkarshmani1997
"""

# Initialization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from ex1_utils import *

## ================ Part 1: Feature Normalization ================

print('Loading data ...','\n')

## Load Data
print('Plotting Data ...','\n')

data = pd.read_csv("ex1data2.txt",names=["size1","bedrooms","price"])
s = np.array(data.size1)
b = np.array(data.bedrooms)
p = np.array(data.price)
m = len(b) # number of training examples

# Design Matrix
s = np.vstack(s)
b = np.vstack(b)
X = np.hstack((s,b))


# Print out some data points
print('First 10 examples from the dataset: \n')
print(" size = ", s[:10],"\n"," bedrooms = ", b[:10], "\n")

input('Program paused. Press enter to continue.\n')

# Scale features to zero mean and standard deviation of 1
print('Normalizing Features ...\n')

X = featureNormalize(X)

# Add intercept term to X
X = np.hstack((np.ones_like(s),X))

## ================ Part 2: Gradient Descent ================

print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.05
num_iters = 400

# Init Theta and Run Gradient Descent 
theta = np.zeros(3)

# Multiple Dimension Gradient Descent
theta, hist = gradientDescentMulti(X, p, theta, alpha, num_iters)

# Plot the convergence graph
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(np.arange(len(hist)),hist ,'-b')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print(theta,'\n')

# Estimating the price of a house on the given size in square feet and no of bedrooms
siz=int(input("Program paused. Enter the size of house in sq-ft:"))
br=int(input("Program paused. Enter the no of bedrooms:"))

# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
normalized_specs = np.array([1,((siz-s.mean())/s.std()),((br-b.mean())/b.std())])
price = np.dot(normalized_specs,theta) 


print('Predicted price of house with size {} and {} no of bedrooms (using gradient descent):\n '.format(siz, br),
      price)

input('Program paused. Press enter to continue.\n')

## ================ Part 3: Normal Equations ================
print('Solving with normal equations...\n')

data = pd.read_csv("ex1data2.txt",names=["sz","bed","price"])
s = np.array(data.sz)
b = np.array(data.bed)
p = np.array(data.price)
m = len(b) # number of training examples

# Design Matrix
s = np.vstack(s)
b = np.vstack(b)
X = np.hstack((s,b))

# Add intercept term to X
X = np.hstack((np.ones_like(s),X))

# Calculate the parameters from the normal equation
theta = normalEqn(X, p)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(theta)
print('\n')

# Estimate the price of a house, by Normal equations
price = np.dot([1,siz,br],theta) # You should change this


print('Predicted price of house with size = {} and {} no of bedrooms (Normalization):\n '.format(siz, br),
      price)


# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
