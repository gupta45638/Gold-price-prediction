#!/usr/bin/env python
# coding: utf-8

# # Gold Price prediction : Code Clause
# Artificial Intellegence and Machine Learning Internship
# 
# Author: Nandini Gupta
# Task 2: Gold Price Prediction
# 
# In this task: We will predict the Gold price in this model 
# There are many steps, which i have followed 
# Step 1: Define the data
# Step 2: Define the hyperparameter
# Step 3: Initialize weights and biases
# Step 4: Initilize the loops
# Step 5: Updated weights and biases
# Step 6: Print the data which is in processing
# Step 7: Print the Predict Gold price
# 
# 
# Here is code.........

# In[14]:


# Data
days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
prices = [1280, 1285, 1290, 1295, 1300, 1290, 1295, 1300, 1305, 1310]

# Hyperparameters
learning_rate = 0.001
epochs = 1000

# Initialize weights and biases
w = 0.0
b = 0.0

# Training loop
i=0
while i<epochs:
    i=i+1
    # Initialize gradients
    dw = 0.0
    db = 0.0
    
    # Forward pass
    j=0
    while j<len(days):
        y_pred = w * days[j] + b
        # Compute gradients
        dw += (y_pred - prices[j]) * days[j]
        db += (y_pred - prices[j])
        j=j+1
    
    # Update weights and biases
    w -= learning_rate * dw / len(days)
    b -= learning_rate * db / len(days)
    
    # Print progress
    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {sum([(w * days[j] + b - prices[j])**2 for j in range(len(days))]) / len(days)}")

# Predict a new gold price
day1 = 11
price1 = w * day1 + b
print(f"Predicted price for day {day1}: {price1}")

