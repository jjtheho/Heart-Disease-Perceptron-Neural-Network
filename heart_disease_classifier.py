# import libraries
import numpy as np
import pandas as pd

# import heart disease data set
data = pd.read_csv('heart.csv')

# Required sigmoid functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

# Cleaning data
data.age = data.age / 10
data.trestbps = data.trestbps / 100
data.chol = data.chol / 100
data.thalach = data.thalach / 100

# Format data as a matrix
df = np.delete(data.values, np.s_[13:14], axis=1)

# Extract expected results from data
target = np.array(data.target)
length = len(target)

expected = []
for i in range(length):
    n = [target[i]]
    expected.append(n)

expected = np.array(expected)

# Generate random weights
np.random.seed(1)

weights = 2 * np.random.random((13, 1)) - 1

# Display weights before training
print('Random synaptic weights before training: ')
print(weights)
print('\n')

# Training the network (iteration = 10000000)
for i in range(10000000):
    input_layer = df

    outputs = sigmoid(np.dot(input_layer, weights))

    error = expected - outputs

    adjustments = error * sigmoid_derivative(outputs)

    weights = weights + np.dot(input_layer.T, adjustments)

# Display weights after training
print('Synaptic Weights after training: ')
print(weights)
print('\n')

# Display results for each patient (303)
# 0 = no heart disease
# 1 = presense of heart disease

count = 0

for i in range(303):

    if (outputs[i] >= 0.9):

        if (expected[i] == 1):
            count = count + 1

        print('Patient ' + str(i + 1) + ': 1')

    elif (outputs[i] <= 0.1):
        if (expected[i] == 0):
            count = count + 1

        print('Patient ' + str(i + 1) + ': 0')

    else:
        print('Patient ' + str(i + 1) + ': ' + str(outputs[i]))

# Display the Accuracy
print('\n')
print('Accuracy: ' + str(float(count * 100) / 303) + '%')
